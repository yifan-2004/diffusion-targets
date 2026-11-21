"""Continue the pixel-space ImageNet-100 DiT-L experiment on one four-GPU node."""
import argparse
import io
import json
import math
import os
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffhq/vendor_dit"))
from models import DiT


TRAIN_SIZE = 126689
VAL_SIZE = 5000
DATASET_REVISION = "f9c6a73853a6d8d834b6e73eec831fa364a83fa3"
STOP_REQUESTED = False


def request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


class CheckpointedDiT(DiT):
    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        conditioning = self.t_embedder(t) + self.y_embedder(y, self.training)
        for block in self.blocks:
            x = checkpoint(block, x, conditioning, use_reentrant=False)
        return self.unpatchify(self.final_layer(x, conditioning))


class ArchivedImageFolder(Dataset):
    """Read the verified, uncompressed RGB archive in ImageFolder order."""

    def __init__(self, archive, split, transform):
        self.archive = archive
        self.transform = transform
        self.fd = None
        prefix = f"rgb256/{split}/"
        index = json.loads(archive.with_suffix(".index.json").read_text())
        assert index["archive_bytes"] == archive.stat().st_size
        entries = index["splits"][split]
        self.classes = sorted({name[len(prefix):].split("/", 1)[0] for name, _, _ in entries})
        class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = [
            (offset, size, class_to_idx[name[len(prefix):].split("/", 1)[0]])
            for name, offset, size in sorted(entries)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.fd is None:
            self.fd = os.open(self.archive, os.O_RDONLY)
        offset, size, label = self.samples[index]
        raw = os.pread(self.fd, size, offset)
        if len(raw) != size:
            raise OSError(f"Incomplete archive read at {offset}: {len(raw)} of {size} bytes")
        with Image.open(io.BytesIO(raw)) as image:
            return self.transform(image.convert("RGB")), label


class MaskedImages(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[max(0, index)]
        return image, label, index >= 0


def rank_batches(length, epoch, batch, rank, world, first_batch):
    order = torch.randperm(length, generator=torch.Generator().manual_seed(2027 + epoch))
    count = math.ceil(length / batch)
    padded = torch.full((count * batch,), -1, dtype=torch.long)
    padded[:length] = order
    per_rank = padded.view(count, batch // world, world)[first_batch:, :, rank]
    return per_rank.tolist()


def losses(model, images, labels, target, rng):
    t = torch.sigmoid(torch.randn(len(images), device=images.device, generator=rng) * 0.8 - 0.8)
    time_image = t[:, None, None, None]
    noise = torch.randn(images.shape, device=images.device, generator=rng)
    mixed = time_image * images + (1 - time_image) * noise
    remaining = (1 - time_image).clamp_min(0.05)
    with torch.autocast("cuda", torch.bfloat16):
        predicted = model(mixed, 1000 * t, labels).float()
    velocity = (predicted - mixed) / remaining if target == "x" else predicted
    return (velocity - (images - mixed) / remaining).square().flatten(1).mean(1)


def validate(model, loader, device, target):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(2029)
    with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        value = torch.cat([
            losses(model, images.to(device), labels.to(device), target, rng)
            for images, labels in loader
        ]).mean().item()
    model.train()
    return value


def save_checkpoint(path, model, optimizer, config, epoch, next_batch):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "weights": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "next_batch": next_batch,
        "dataset": DATASET_REVISION,
    }, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epoch1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch", type=int, choices=(2, 4, 8, 16), required=True)
    parser.add_argument("--target", choices=("x", "v"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-minutes", type=float, default=0)
    parser.add_argument("--max-updates", type=int, default=0)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4
    assert args.batch > 32 and args.batch % (world * args.microbatch) == 0
    assert args.save_every > 0 and args.epochs >= 2
    torch.manual_seed(2026)

    transform = transforms.Compose([
        transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    if args.data.is_file():
        images = ArchivedImageFolder(args.data, "train", transform)
        validation = ArchivedImageFolder(args.data, "validation", transform)
    else:
        images = ImageFolder(args.data / "train", transform)
        validation = ImageFolder(args.data / "validation", transform)
    assert len(images) == TRAIN_SIZE and len(validation) == VAL_SIZE
    assert len(images.classes) == 100 and images.classes == validation.classes
    train_data = MaskedImages(images)
    indices = torch.randperm(len(validation), generator=torch.Generator().manual_seed(2030))[:32].tolist()
    val_loader = DataLoader(Subset(validation, indices), batch_size=1)
    updates_per_epoch = math.ceil(TRAIN_SIZE / args.batch)

    model_class = DiT if args.patch == 16 else CheckpointedDiT
    model = model_class(input_size=256, patch_size=args.patch, in_channels=3,
                        hidden_size=1024, depth=24, num_heads=16, num_classes=100,
                        learn_sigma=False).to(device)
    assert all(block.attn.fused_attn for block in model.blocks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, fused=True)
    config = dict(size="L", patch=args.patch, target=args.target, batch=args.batch,
                  microbatch=args.microbatch, world_size=world, precision="bf16")
    args.output.mkdir(parents=True, exist_ok=True)
    last = args.output / "last.pt"
    epoch, next_batch = 2, 0
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=True, mmap=True)
        assert state["config"] == config and state["dataset"] == DATASET_REVISION
        model.load_state_dict(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, next_batch = state["epoch"], state["next_batch"]
    else:
        state = torch.load(args.epoch1, map_location="cpu", weights_only=True, mmap=True)
        cfg = state["config"]
        assert tuple(cfg[k] for k in ("size", "patch", "target", "steps", "batch")) == (
            "L", args.patch, args.target, 3960, 32
        )
        assert state["dataset"] == DATASET_REVISION
        model.load_state_dict(state["weights"])
    del state
    ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    if rank == 0:
        print(json.dumps(dict(config, starting_epoch=epoch, starting_batch=next_batch,
                              optimizer_restored=last.is_file())), flush=True)
    status = torch.zeros((), device=device, dtype=torch.int32)
    deadline = time.monotonic() + 60 * args.max_minutes if args.max_minutes else math.inf
    rng = torch.Generator(device=device)
    updates_this_run = 0
    last_update = time.perf_counter()
    for current_epoch in range(epoch, args.epochs + 1):
        batch_sampler = rank_batches(TRAIN_SIZE, current_epoch, args.batch, rank, world, next_batch)
        loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
        for batch_index, (image_batch, label_batch, valid_batch) in enumerate(loader, next_batch):
            start = time.perf_counter()
            step = 3960 + (current_epoch - 2) * updates_per_epoch + batch_index + 1
            rng.manual_seed(1_000_000 + step * world + rank)
            torch.cuda.manual_seed(2_000_000 + step * world + rank)
            valid_count = min(args.batch, TRAIN_SIZE - batch_index * args.batch)
            optimizer.zero_grad(set_to_none=True)
            local_total = torch.zeros((), device=device)
            splits = list(zip(image_batch.split(args.microbatch), label_batch.split(args.microbatch),
                              valid_batch.split(args.microbatch)))
            for micro_index, (micro_images, labels, valid) in enumerate(splits):
                synchronization = ddp.no_sync() if micro_index + 1 < len(splits) else nullcontext()
                with synchronization, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    per_image = losses(ddp, micro_images.to(device, non_blocking=True),
                                       labels.to(device, non_blocking=True), args.target, rng)
                    masked = (per_image * valid.to(device, non_blocking=True)).sum()
                    (masked * (world / valid_count)).backward()
                local_total += masked.detach()
            dist.all_reduce(local_total)
            value = local_total.item() / valid_count
            if not math.isfinite(value):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            optimizer.step()
            updates_this_run += 1
            end_epoch = batch_index + 1 == updates_per_epoch
            images_seen = TRAIN_SIZE * (current_epoch - 1) + min(
                (batch_index + 1) * args.batch, TRAIN_SIZE
            )
            if rank == 0 and (batch_index == 0 or step % 20 == 0 or end_epoch):
                torch.cuda.synchronize()
                print(json.dumps(dict(epoch=current_epoch, step=step, images_seen=images_seen,
                                      loss=value, seconds=time.perf_counter() - start,
                                      wall_seconds=time.perf_counter() - last_update)), flush=True)
            torch.cuda.synchronize()
            last_update = time.perf_counter()
            if step % 100 == 0 or end_epoch or updates_this_run == args.max_updates:
                if rank == 0:
                    value = validate(model, val_loader, device, args.target)
                    print(json.dumps(dict(epoch=current_epoch, step=step, images_seen=images_seen,
                                          validation_loss=value)), flush=True)
                dist.barrier()
            status.fill_(int(STOP_REQUESTED or time.monotonic() >= deadline or
                             (args.max_updates and updates_this_run >= args.max_updates)))
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
            stop = bool(status.item())
            if end_epoch or step % args.save_every == 0 or stop:
                if rank == 0:
                    save_checkpoint(last, model, optimizer, config,
                                    current_epoch + int(end_epoch), 0 if end_epoch else batch_index + 1)
                dist.barrier()
            if stop:
                dist.destroy_process_group()
                return
        next_batch = 0

    if rank == 0:
        (args.output / "TRAINING_COMPLETE").write_text(f"epoch {args.epochs}\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
