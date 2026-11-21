"""Prepare the shared COCO split, normalization and training-image PCA."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from plot_pca_three_targets import patches

ROOT = Path(__file__).resolve().parent


def prepare(source):
    cache = ROOT / 'coco_data/coco_val2017_preprocessed_32.npz'
    cache.parent.mkdir(exist_ok=True)
    if source.is_dir():
        files = sorted(source.glob('*.jpg'))
        if len(files) != 5000:
            raise ValueError('Expected all 5000 val2017 JPEGs')
        pixels = []
        for file in files:
            with Image.open(file) as im:
                pixels.append(np.asarray(ImageOps.fit(im.convert('RGB'), (32, 32), method=Image.Resampling.BICUBIC)))
        np.savez(cache, images=np.stack(pixels), names=[f.name for f in files])
    elif source.resolve() != cache.resolve():
        shutil.copyfile(source, cache)
    with np.load(cache, allow_pickle=False) as data:
        names = data['names'].tolist()
        pixels = data['images']
    if pixels.shape != (5000,32,32,3) or pixels.dtype != np.uint8 or names != sorted(set(names)):
        raise ValueError('Expected the sorted RGB val2017 cache')
    images = torch.from_numpy(pixels.astype(np.float32)/127.5-1).permute(0,3,1,2).contiguous()
    perm = torch.randperm(len(images), generator=torch.Generator().manual_seed(20260926))
    held, train = images[perm[:1000]], images[perm[1000:]]
    out = ROOT / 'prepared'
    out.mkdir(exist_ok=True)
    split = {'train':[names[i] for i in perm[1000:]], 'heldout':[names[i] for i in perm[:1000]]}
    (out/'split.json').write_text(json.dumps(split))
    mean = train.mean(0, keepdim=True)
    train = train-mean
    scale = train.square().mean().sqrt()
    train /= scale
    held = (held-mean)/scale
    torch.save(dict(train_mean=mean, scale=scale), out/'data_summary.pt')
    p = patches(train).double()
    mu = p.mean(0)
    p -= mu
    lam, u = torch.linalg.eigh(p.T@p/len(p))
    h = patches(held).double()-mu
    state = dict(mean=mu, eigenvalues=lam.flip(0), eigenvectors=u.flip(1), heldout_second_moment=h.T@h/len(h))
    torch.save(state, out/'pca_state.pt')
    torch.save(dict(train=train,heldout=held,pca=state), out/'width_data.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='COCO val2017 JPEG directory or original lossless NPZ cache')
    prepare(parser.parse_args().source)
