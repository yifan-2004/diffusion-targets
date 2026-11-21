"""Generate matched FFHQ samples and calculate FID-1k for a paired checkpoint."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch_fidelity import calculate_metrics

from run_pair import device_choice, make_model, predicted_velocity


def save_png(x, path):
    a = ((x.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5)
    Image.fromarray(np.rint(a).astype(np.uint8)).save(path)


@torch.no_grad()
def sample(model, arm, z, steps, t_eps):
    y = torch.zeros(len(z), dtype=torch.long, device=z.device)

    def velocity(state, t_scalar):
        t = torch.full((len(state),), t_scalar, device=state.device)
        d = (1 - t[:, None, None, None]).clamp_min(t_eps)
        out = model(state, t * 1000, y)
        return predicted_velocity(out, arm, state, d)

    for i in range(steps):
        t0, t1 = i / steps, (i + 1) / steps
        v0 = velocity(z, t0)
        if i == steps - 1:
            z = z + (t1 - t0) * v0
        else:
            predictor = z + (t1 - t0) * v0
            v1 = velocity(predictor, t1)
            z = z + (t1 - t0) * 0.5 * (v0 + v1)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--count', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--skip-fid', action='store_true')
    args = ap.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = argparse.Namespace(**state['args'])
    split = json.loads((cfg.data / 'split.json').read_text())
    if args.count > len(split['validation']):
        raise ValueError('FID reference count exceeds held-out images')
    device = device_choice()
    outdir = args.checkpoint.parent / f'eval_step{state["step"]}_heun{args.steps}_n{args.count}'
    outdir.mkdir(exist_ok=True)
    refdir = cfg.data / f'fid_reference_{args.count}'
    refdir.mkdir(exist_ok=True)
    for i, name in enumerate(split['validation'][:args.count]):
        dest = refdir / f'{i:05d}.png'
        if not dest.exists():
            shutil.copyfile(cfg.data / name, dest)
    results = {}
    for arm in ('x', 'v'):
        model = make_model(cfg, device)
        model.load_state_dict(state['models'][arm])
        model.eval()
        samples = outdir / arm
        samples.mkdir(exist_ok=True)
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        for offset in range(0, args.count, args.batch):
            b = min(args.batch, args.count - offset)
            z = torch.randn(b, 3, 128, 128, generator=generator).to(device) * cfg.noise_scale
            images = sample(model, arm, z, args.steps, cfg.t_eps)
            for j, image in enumerate(images):
                save_png(image, samples / f'{offset + j:05d}.png')
            if (offset + b) % 100 == 0:
                print(json.dumps({'arm': arm, 'generated': offset + b}), flush=True)
        result = {'step': state['step'], 'arm': arm, 'count': args.count,
                  'heun_steps': args.steps, 'seed': args.seed,
                  'reference': str(refdir), 'samples': str(samples)}
        if not args.skip_fid:
            metrics = calculate_metrics(input1=str(samples), input2=str(refdir),
                                        cuda=False, fid=True, isc=False, kid=False,
                                        verbose=False, save_cpu_ram=True)
            result['fid_1k'] = float(metrics['frechet_inception_distance'])
        results[arm] = result
        print(json.dumps(result), flush=True)
    (outdir / 'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
