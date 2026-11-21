"""Pixel-space, class-conditioned ImageNet-100 DiT pilot."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import mse_loss
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffhq/vendor_dit"))
from models import DiT

SIZES = {"S": (384, 12, 6), "B": (768, 12, 12), "L": (1024, 24, 16), "XL": (1152, 28, 16)}


class CheckpointedDiT(DiT):
    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        conditioning = self.t_embedder(t) + self.y_embedder(y, self.training)
        for block in self.blocks:
            x = checkpoint(block, x, conditioning, use_reentrant=False)
        return self.unpatchify(self.final_layer(x, conditioning))


def flow_loss(model, images, labels, target, generator):
    t = torch.sigmoid(torch.randn(len(images), device=images.device, generator=generator) * 0.8 - 0.8)
    time = t[:, None, None, None]
    noise = torch.randn(images.shape, device=images.device, generator=generator)
    mixed = time * images + (1 - time) * noise
    remaining = (1 - time).clamp_min(0.05)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast("cuda", torch.bfloat16):
        prediction = model(mixed, 1000 * t, labels)
    velocity = (prediction.float() - mixed) / remaining if target == "x" else prediction.float()
    return mse_loss(velocity, (images - mixed) / remaining)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--patch", type=int, choices=(2, 4, 8, 16), required=True)
    parser.add_argument("--target", choices=("x", "v"), default="x")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--data", default="rgb256", help="Folder containing train/ and validation/")
    args = parser.parse_args()
    assert args.batch % args.microbatch == 0
    torch.manual_seed(2026)
    width, depth, heads = SIZES[args.size]
    model = CheckpointedDiT(
        input_size=256, patch_size=args.patch, in_channels=3, hidden_size=width,
        depth=depth, num_heads=heads, num_classes=100, learn_sigma=False,
    ).cuda()
    assert all(block.attn.fused_attn for block in model.blocks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, fused=True)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rng = torch.Generator(device="cuda").manual_seed(2027)
    if args.probe:
        args.steps = 4
        images = torch.randn(args.microbatch, 3, 256, 256, device="cuda")
        labels = torch.zeros(args.microbatch, device="cuda", dtype=torch.long)
    else:
        data = Path(args.data)
        transform = transforms.Compose([
            transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        dataset = ImageFolder(data / "train", transform)
        assert len(dataset) == 126689 and len(dataset.classes) == 100
        batches = DataLoader(dataset, batch_size=args.microbatch, shuffle=True, num_workers=4,
                                 pin_memory=True, generator=torch.Generator().manual_seed(2028))
        loader = iter(batches)
        validation = ImageFolder(data / "validation", transform)
        assert len(validation) == 5000 and validation.classes == dataset.classes
        indices = torch.randperm(len(validation), generator=torch.Generator().manual_seed(2030))[:32].tolist()
        validation_loader = DataLoader(Subset(validation, indices), batch_size=1)

    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        if args.probe:
            torch.cuda.synchronize()
        started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        microsteps = args.batch // args.microbatch
        total = torch.zeros((), device="cuda")
        for _ in range(microsteps):
            if not args.probe:
                try:
                    images, labels = next(loader)
                except StopIteration:
                    loader = iter(batches)
                    images, labels = next(loader)
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            loss = flow_loss(model, images, labels, args.target, rng) / microsteps
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                loss.backward()
            total += loss.detach()
        optimizer.step()
        value = total.item()
        if not torch.isfinite(total).item():
            raise RuntimeError(f"Nonfinite loss at step {step}")
        if args.probe or step == 1 or step % 20 == 0:
            print(json.dumps({"step": step, "loss": value, "parameters": params,
                              "seconds": perf_counter() - started, "batch": args.batch,
                              "microbatch": args.microbatch,
                              "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                              "reserved_gib": torch.cuda.max_memory_reserved() / 2**30}), flush=True)
        if not args.probe and (step % 100 == 0 or step == args.steps):
            model.eval()
            validation_rng = torch.Generator(device="cuda").manual_seed(2029)
            with torch.no_grad():
                losses = [flow_loss(model, images.cuda(), labels.cuda(), args.target, validation_rng)
                          for images, labels in validation_loader]
            print(json.dumps({"step": step, "validation_loss": torch.stack(losses).mean().item()}), flush=True)
            model.train()
    if not args.probe:
        path = Path(f"{args.size}_p{args.patch}_{args.target}.pt")
        temporary = path.with_suffix(".tmp")
        weights = {k: v.detach().to(device="cpu", dtype=torch.bfloat16) for k, v in model.state_dict().items()}
        torch.save({"weights": weights, "config": vars(args), "dataset": "f9c6a73853a6d8d834b6e73eec831fa364a83fa3"}, temporary)
        os.replace(temporary, path)


if __name__ == "__main__":
    main()
