"""Extract the verified 128px FFHQ Arrow shard and write an immutable split."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arrow', type=Path, default=Path(__file__).parent / 'data/ffhq128_shard0.arrow')
    ap.add_argument('--output', type=Path, default=Path(__file__).parent / 'data/ffhq128_shard0')
    ap.add_argument('--train-count', type=int, default=12000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = []
    with pa.memory_map(str(args.arrow), 'r') as source:
        reader = pa.ipc.open_stream(source)
        for batch in reader:
            for row in batch.column(0).to_pylist():
                name, data = row['path'], row['bytes']
                if not name.endswith('.png') or '/' in name or '..' in name:
                    raise ValueError(f'Unexpected image name: {name}')
                with Image.open(io.BytesIO(data)) as im:
                    if im.size != (128, 128) or im.mode != 'RGB':
                        raise ValueError(f'Unexpected image properties: {name} {im.size} {im.mode}')
                dest = args.output / name
                if not dest.exists() or dest.stat().st_size != len(data):
                    dest.write_bytes(data)
                names.append(name)
    if not 0 < args.train_count < len(names) or len(names) != len(set(names)):
        raise ValueError('Invalid FFHQ split or repeated image name')
    names.sort()
    manifest = {
        'source': 'https://huggingface.co/datasets/luethan2025/FFHQ-128x128',
        'original_dataset': 'https://github.com/NVlabs/ffhq-dataset',
        'source_note': 'Unofficial mirror of the official 128x128 thumbnails; pixel identity has not been cross-checked against NVIDIA metadata.',
        'arrow_sha256': hashlib.sha256(args.arrow.read_bytes()).hexdigest(),
        'resolution': 128,
        'train': names[:args.train_count],
        'validation': names[args.train_count:],
    }
    (args.output / 'split.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'train': len(manifest['train']), 'validation': len(manifest['validation']), 'path': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
