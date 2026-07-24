"""
src/transfer/plot_training_curves.py
---------------------------------------
Plots train/val loss and accuracy curves for any set of runs side by
side -- generalized so it works for frozen-vs-finetuned, finetuned-vs-
regularized, finetuned-vs-attention, or any other comparison, not just
one hardcoded pair. Reads training_history.csv that train.py already
wrote -- doesn't need torch, doesn't re-run anything.

Usage (default: frozen vs finetuned):
    python src/transfer/plot_training_curves.py

Usage (custom comparison, e.g. finetuned vs regularized):
    python src/transfer/plot_training_curves.py \
        --dirs outputs/transfer/resnet18_pretrained_finetuned outputs/transfer/resnet18_pretrained_finetuned_regularized_extra \
        --labels Finetuned Finetuned+Regularized \
        --out_file outputs/transfer/training_curves_finetuned_vs_regularized.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_DIRS = [
    "outputs/transfer/resnet18_pretrained_frozen",
    "outputs/transfer/resnet18_pretrained_finetuned",
]
DEFAULT_LABELS = ["Frozen", "Finetuned"]
PALETTE = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]  # cycles if >5 runs


def read_history(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    return {
        "epoch": [int(r["epoch"]) for r in rows],
        "train_loss": [float(r["train_loss"]) for r in rows],
        "train_acc": [float(r["train_acc"]) for r in rows],
        "val_loss": [float(r["val_loss"]) for r in rows],
        "val_acc": [float(r["val_acc"]) for r in rows],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", default=None,
                         help="Method output directories to compare (each must contain "
                              "training_history.csv). Defaults to frozen vs finetuned.")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Legend labels, same order/length as --dirs. Defaults to the "
                              "directory names if not given.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out_file", default=None)
    args = parser.parse_args()

    dirs = args.dirs if args.dirs else DEFAULT_DIRS
    if args.labels:
        if len(args.labels) != len(dirs):
            raise ValueError(f"--labels has {len(args.labels)} entries but --dirs has {len(dirs)}")
        labels = args.labels
    elif args.dirs is None:
        labels = DEFAULT_LABELS
    else:
        labels = [Path(d).name for d in dirs]

    histories = {}
    for d, label in zip(dirs, labels):
        history_path = Path(d) / "training_history.csv"
        if not history_path.exists():
            print(f"WARNING: {history_path} not found, skipping '{label}'.")
            continue
        histories[label] = read_history(history_path)

    if not histories:
        raise FileNotFoundError("No training_history.csv files found for the given --dirs.")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    for i, (label, h) in enumerate(histories.items()):
        color = PALETTE[i % len(PALETTE)]
        ax_loss.plot(h["epoch"], h["train_loss"], color=color, linestyle="--",
                     label=f"{label} (train)")
        ax_loss.plot(h["epoch"], h["val_loss"], color=color, linestyle="-",
                     label=f"{label} (val)")

        ax_acc.plot(h["epoch"], h["train_acc"], color=color, linestyle="--",
                    label=f"{label} (train)")
        ax_acc.plot(h["epoch"], h["val_acc"], color=color, linestyle="-",
                    label=f"{label} (val)")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Training/validation loss")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Training/validation accuracy")
    ax_acc.legend(fontsize=9)
    ax_acc.grid(alpha=0.3)

    fig.suptitle(args.title or " vs ".join(histories.keys()) + ": training curves")
    plt.tight_layout()

    if args.out_file:
        out_file = Path(args.out_file)
    else:
        suffix = "_vs_".join(l.lower().replace(" ", "_").replace("+", "") for l in histories.keys())
        out_file = Path("outputs/transfer") / f"training_curves_{suffix}.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
