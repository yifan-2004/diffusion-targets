"""Paired pixel-DiT x/v flow-matching experiment on a fixed FFHQ split.

The backbone is Meta's unmodified DiT-S (vendor_dit/models.py). The experiment
adapts its input/output to RGB pixels and uses rectified flow instead of DDPM.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / 'vendor_dit'))
from models import DiT  # noqa: E402


STOP = False


def on_signal(_signum, _frame):
    global STOP
    STOP = True


def device_choice():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def make_model(args, device):
    torch.manual_seed(args.model_seed)
    model = DiT(input_size=128, patch_size=args.patch, in_channels=3,
                hidden_size=args.hidden, depth=args.depth, num_heads=args.heads,
                mlp_ratio=4, class_dropout_prob=0, num_classes=1,
                learn_sigma=False)
    return model.to(device)


def initial_hash(model):
    h = hashlib.sha256()
    for v in model.state_dict().values():
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_images(root, names, indices, device):
    arrs = []
    for i in indices:
        with Image.open(root / names[int(i)]) as im:
            if im.size != (128, 128) or im.mode != 'RGB':
                raise ValueError(f'Unexpected FFHQ image: {names[int(i)]}')
            arrs.append(np.asarray(im).copy())
    arr = np.stack(arrs)
    return torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)


def draw_corruption(x, step, micro_index, args):
    # Paired arms receive exactly the same x, t, and epsilon at each update.
    torch.manual_seed(args.noise_seed + step * 1009 + micro_index)
    t = torch.sigmoid(torch.randn(len(x), device=x.device) * args.t_std + args.t_mean)
    tt = t[:, None, None, None]
    eps = torch.randn_like(x) * args.noise_scale
    z = tt * x + (1 - tt) * eps
    d = (1 - tt).clamp_min(args.t_eps)
    target = (x - z) / d
    return z, t, target, d


def predicted_velocity(net_output, target_type, z, d):
    if target_type == 'x':
        return (net_output - z) / d
    return net_output


def append_csv(path, row):
    fresh = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def save_checkpoint(path, models, opts, step, train_seconds, args, init_hash, arm_steps):
    tmp = path.with_suffix('.tmp')
    torch.save({'models': {a: m.state_dict() for a, m in models.items()},
                'optimizers': {a: o.state_dict() for a, o in opts.items()},
                'step': step, 'train_seconds': train_seconds,
                'args': vars(args), 'initial_hash': init_hash, 'arm_steps': arm_steps}, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(models, root, names, args, device, step):
    for m in models.values():
        m.eval()
    results = {a: {'sum': 0.0, 't_low': 0.0, 't_mid': 0.0, 't_high': 0.0} for a in models}
    count = min(args.val_count, len(names))
    for offset in range(0, count, args.microbatch):
        idx = range(offset, min(offset + args.microbatch, count))
        x = load_images(root, names, idx, device)
        for ti, t_value in enumerate((0.2, 0.5, 0.8)):
            torch.manual_seed(args.val_seed + offset * 17 + ti)
            eps = torch.randn_like(x) * args.noise_scale
            t = torch.full((len(x),), t_value, device=device)
            tt = t[:, None, None, None]
            z = tt * x + (1 - tt) * eps
            d = (1 - tt).clamp_min(args.t_eps)
            target = (x - z) / d
            for arm, model in models.items():
                pred = model(z, t * 1000, torch.zeros(len(x), dtype=torch.long, device=device))
                mse = F.mse_loss(predicted_velocity(pred, arm, z, d), target).item()
                results[arm][('t_low', 't_mid', 't_high')[ti]] += mse * len(x)
    rows = {}
    for arm in models:
        vals = {k: results[arm][k] / count for k in ('t_low', 't_mid', 't_high')}
        rows[arm] = {'step': step, 'images_seen': step * args.batch,
                     'arm': arm, 'val_v_mse': sum(vals.values()) / 3, **vals}
    for m in models.values():
        m.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path(__file__).parent / 'data/ffhq128_shard0')
    ap.add_argument('--output', type=Path, default=Path(__file__).parent / 'runs/p8')
    ap.add_argument('--patch', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=384)
    ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--batch', type=int, default=16, help='Effective batch, identical for all patches')
    ap.add_argument('--microbatch', type=int, default=2)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--noise-scale', type=float, default=1.0)
    ap.add_argument('--t-mean', type=float, default=-0.8)
    ap.add_argument('--t-std', type=float, default=0.8)
    ap.add_argument('--t-eps', type=float, default=0.05)
    ap.add_argument('--model-seed', type=int, default=1234)
    ap.add_argument('--data-seed', type=int, default=4321)
    ap.add_argument('--noise-seed', type=int, default=8421)
    ap.add_argument('--val-seed', type=int, default=9361)
    ap.add_argument('--val-count', type=int, default=64)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--save-every', type=int, default=100)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--train-arms', nargs='+', choices=('x', 'v'), default=['x', 'v'],
                    help='Only update selected arms; keep unselected model/optimizer states unchanged')
    args = ap.parse_args()
    if 128 % args.patch or args.batch % args.microbatch or args.hidden % args.heads:
        ap.error('patch must divide 128, microbatch must divide batch, heads must divide hidden')
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    split = json.loads((args.data / 'split.json').read_text())
    train_names, val_names = split['train'], split['validation']
    args.output.mkdir(parents=True, exist_ok=True)
    device = device_choice()
    models = {arm: make_model(args, device) for arm in ('x', 'v')}
    hashes = {a: initial_hash(m) for a, m in models.items()}
    assert hashes['x'] == hashes['v'], 'Pair must have identical initial weights'
    opts = {a: torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            for a, m in models.items()}
    params = sum(p.numel() for p in models['x'].parameters())
    protocol = {'dataset': split['source'], 'subset_train': len(train_names),
                'subset_validation': len(val_names), 'source_sha256': split['arrow_sha256'],
                'architecture': 'Meta DiT-S source ed81ce2229091fd4ecc9a223645f95cf379d582b, pixel RGB, learn_sigma=False, no RoPE',
                'parameter_count_per_arm': params, 'patch': args.patch,
                'token_count': (128 // args.patch) ** 2, 'patch_output_dim': 3 * args.patch ** 2,
                'initial_hash': hashes['x'], 'objective': 'same clipped velocity target for both arms',
                'time': 'z=t*x+(1-t)*eps; t=sigmoid(N(-0.8,0.8^2)); t=0 noise; t=1 image',
                'timestep_embedding': '1000*t for official DiT sinusoidal embedder',
                'pairing': 'same image indices, t and epsilon at each update; same initialization',
                'args': vars(args), 'device': str(device), 'torch': torch.__version__}
    protocol_path = args.output / 'protocol.json'
    checkpoint = args.output / 'checkpoint-last.pt'
    step = 0
    arm_steps = {a: 0 for a in models}
    train_seconds = 0.0
    if args.resume:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        old = state['args']
        for k in ('patch', 'hidden', 'depth', 'heads', 'batch', 'microbatch', 'lr', 'data_seed', 'noise_seed'):
            if old[k] != vars(args)[k]:
                raise ValueError(f'Cannot resume with changed {k}')
        for arm in models:
            models[arm].load_state_dict(state['models'][arm])
            opts[arm].load_state_dict(state['optimizers'][arm])
        step, train_seconds = state['step'], state['train_seconds']
        arm_steps = {a: state.get('arm_steps', {}).get(a, step) for a in models}
    elif checkpoint.exists() or protocol_path.exists():
        raise FileExistsError(f'Output already contains a run: {args.output}')
    for arm, model in models.items():
        if arm not in args.train_arms:
            model.requires_grad_(False)
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False, default=str))
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    status_path = args.output / 'status.json'
    print(json.dumps({'event': 'start', 'step': step, 'patch': args.patch, 'device': str(device),
                      'parameters_per_arm': params}), flush=True)
    status_path.write_text(json.dumps({'status': 'running', 'step': step, 'train_seconds': train_seconds, 'arm_steps': arm_steps, 'train_arms': args.train_arms}))
    try:
        while step < args.steps and not STOP:
            step_start = time.monotonic()
            indices = np.random.default_rng(args.data_seed + step).integers(0, len(train_names), size=args.batch)
            losses = {a: (0.0 if a in args.train_arms else None) for a in models}
            for optimizer in opts.values():
                optimizer.zero_grad(set_to_none=True)
            for micro_index, offset in enumerate(range(0, args.batch, args.microbatch)):
                x = load_images(args.data, train_names, indices[offset:offset + args.microbatch], device)
                z, t, target, d = draw_corruption(x, step, micro_index, args)
                y = torch.zeros(len(x), dtype=torch.long, device=device)
                for arm, model in models.items():
                    if arm not in args.train_arms:
                        continue
                    out = model(z, t * 1000, y)
                    loss = F.mse_loss(predicted_velocity(out, arm, z, d), target)
                    if not bool(torch.isfinite(loss).item()):
                        raise RuntimeError(f'Nonfinite {arm} loss at step {step}')
                    (loss * (len(x) / args.batch)).backward()
                    losses[arm] += loss.detach().item() * (len(x) / args.batch)
            for arm in args.train_arms:
                torch.nn.utils.clip_grad_norm_(models[arm].parameters(), 1.0)
                opts[arm].step()
            step += 1
            for arm in args.train_arms:
                arm_steps[arm] = step
            train_seconds += time.monotonic() - step_start
            if step == 1 or step % args.log_every == 0:
                row = {'step': step, 'images_seen': step * args.batch,
                       'train_seconds': round(train_seconds, 3),
                       'train_x_v_mse': losses['x'], 'train_v_v_mse': losses['v']}
                append_csv(args.output / 'train.csv', row)
                print(json.dumps(row), flush=True)
            if step % args.eval_every == 0:
                rows = validate(models, args.data, val_names, args, device, step)
                for row in rows.values():
                    row['train_seconds'] = round(train_seconds, 3)
                    append_csv(args.output / 'validation.csv', row)
                print(json.dumps({'event': 'validation', 'rows': rows}), flush=True)
            if step % args.save_every == 0 or STOP:
                save_checkpoint(checkpoint, models, opts, step, train_seconds, args, hashes['x'], arm_steps)
                status_path.write_text(json.dumps({'status': 'running' if not STOP else 'stopped',
                                                   'step': step, 'train_seconds': train_seconds, 'arm_steps': arm_steps, 'train_arms': args.train_arms}))
        save_checkpoint(checkpoint, models, opts, step, train_seconds, args, hashes['x'], arm_steps)
        status_path.write_text(json.dumps({'status': 'completed' if step >= args.steps else 'stopped',
                                           'step': step, 'train_seconds': train_seconds, 'arm_steps': arm_steps, 'train_arms': args.train_arms}))
    except Exception as e:
        status_path.write_text(json.dumps({'status': 'failed', 'step': step, 'error': str(e)}))
        raise


if __name__ == '__main__':
    main()
