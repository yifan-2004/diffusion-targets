"""Render recorded losses and export one CSV per experiment."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
curves = root / "curves"
curves.mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
for patch, ax in zip((2, 4, 8, 16), axes.flat):
    for target, color in (("x", "#195d9d"), ("v", "#a94528")):
        arm = f"L_p{patch}_{target}"
        records = {}
        for log in (root.parent / "epoch1" / "logs" / f"{arm}.initial.jsonl",
                    root.parent / "epoch1" / "logs" / f"{arm}.jsonl",
                    root / arm / "training.jsonl"):
            if not log.exists():
                continue
            for line in log.read_text().splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if "step" not in row:
                    continue
                for metric in ("loss", "validation_loss"):
                    if metric in row:
                        step = row["step"]
                        images = row.get("images_seen", min(step * 32, 126689))
                        records[(step, metric)] = (step, images, metric, row[metric])
        with (curves / f"{arm}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("step", "images_seen", "metric", "value"))
            writer.writerows(row for _, row in sorted(records.items()))
        for metric, style, label in (("loss", "-", "train"), ("validation_loss", "--", "val")):
            values = [row for _, row in sorted(records.items()) if row[2] == metric]
            if values:
                ax.plot([row[1] / 126689 for row in values], [row[3] for row in values],
                        linestyle=style, color=color, label=f"{target} {label}", linewidth=1.2)
    ax.axvline(1, color="0.7", linewidth=0.8)
    ax.set(title=f"Patch size {patch}", xlabel="Epochs of data processed", ylabel="Velocity MSE")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)
fig.savefig(curves / "loss_curves.png", dpi=200)
fig.savefig(curves / "loss_curves.pdf")
