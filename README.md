# Diffusion prediction targets

Training and analysis code accompanying *How Image Energy Distribution and Information Bottlenecks Affect Prediction Target Selection in Diffusion Models*. Commands below assume a Linux or macOS shell, Python 3.12, and the repository root as the working directory. Install the PyTorch build for your accelerator, then run `python -m pip install -r requirements.txt`. Supply the datasets separately. The vendored DiT backbone is from [Meta DiT](https://github.com/facebookresearch/DiT/tree/ed81ce2229091fd4ecc9a223645f95cf379d582b) under its included CC BY-NC 4.0 license.

## Experiment 1 · COCO patch energy

Prepare all 5,000 images in COCO val2017. The script resizes them to 32 × 32, reconstructs the seeded 4,000/1,000 split, and fits the image PCA basis on training patches. The diagnostic projects image, noise and velocity into this same basis.

```sh
python coco/prepare.py /path/to/val2017
python coco/plot_pca_three_targets.py
```

Per-direction and cumulative energies are in `coco/coco_pca_three_targets/diagnostics.json`.

## Experiment 2 · Controlled training

Part A varies the TinyJiT input bottleneck. Run Experiment 1 preparation first. Each invocation writes a curve, final validation results and weights. Completed arms are skipped on rerun.

```sh
for width in $(seq 2 2 50); do
  for head in x eps v; do
    for seed in 0 1 2; do
      python coco/pca_parallel_worker.py --data coco/prepared/width_data.pt --output runs/width --bottleneck "$width" --head "$head" --seed "$seed"
    done
  done
done
```

Part B uses a second ViT at fixed capacity. The trainer constructs four synthetic distributions and one real COCO condition, then fits image and noise predictors to the same 50/50 mixture. Training logs and weights are written under `energy_vit/runs/`. Restart with the same command to resume.

```sh
python energy_vit/run.py
```

## Experiment 3 · FFHQ pixel DiT

The available paired FFHQ runner trains DiT-S with a shared velocity loss, logs training and validation errors, and saves both arms in one checkpoint. Its included preparation script reads a single Arrow shard. This is the local single-device implementation. The full-data four-GPU FFHQ launch scripts underlying the configuration in the paper have not been verified for this release. The following is a runnable example on the shard.

```sh
python ffhq/prepare_ffhq.py --arrow /path/to/ffhq128_shard0.arrow
python ffhq/run_pair.py --patch 8 --batch 32 --microbatch 2 --steps 2000 --output runs/p8
python ffhq/evaluate_pair.py --checkpoint runs/p8/checkpoint-last.pt --steps 100 --count 1000
```

Use a separate output directory for each patch. Pass `--resume` with the original configuration to continue a run. Evaluation uses the dataset path recorded in the checkpoint.

## Subsequent ImageNet-100 training

`imagenet_continuation/` contains the archived DiT-L pixel-space trainer and four-GPU continuation entry point. These scripts use the **full** 126,689-image training split and 5,000 validation images of `basavyr/imagenet-100` (revision `f9c6a73853a6d8d834b6e73eec831fa364a83fa3`). Supply the original 256 × 256 RGB class directories. The four-GPU entry point requires CUDA with BF16 and flash attention. The example starts with one GPU and continues on four. Use separate checkpoints and output directories for each patch and target.

```sh
python imagenet_continuation/train_epoch1.py --data /path/to/rgb256 --size L --patch 4 --target x --steps 3960 --batch 32 --microbatch 1
torchrun --standalone --nproc-per-node=4 imagenet_continuation/train_ddp.py --data /path/to/rgb256 --epoch1 L_p4_x.pt --output runs/L_p4_x --patch 4 --target x --epochs 10 --batch 64 --microbatch 1
```

The first continuation initializes from the epoch-one weights. Subsequent restarts from the same output directory also restore optimizer state and batch position.
