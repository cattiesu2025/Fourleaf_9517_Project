"""
src/transfer/compute_metrics.py
----------------------------------
Computes the metrics the spec requires (top-1, top-5 accuracy, macro
precision/recall/F1) from the predictions.csv + scores.npz that
predict.py already wrote. Doesn't re-run inference -- just reads the
saved scores.

Usage (single method):
    python src/transfer/compute_metrics.py --strategy finetuned

Usage (compare both official methods side by side):
    python src/transfer/compute_metrics.py --compare
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer.model import STRATEGY_TO_METHOD_NAME


def compute_metrics_for_method(output_dir: Path) -> dict:
    scores_path = output_dir / "scores.npz"
    pred_path = output_dir / "predictions.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} not found -- run predict.py first.")

    data = np.load(scores_path, allow_pickle=True)
    scores = data["scores"]              # (N, num_classes) softmax probabilities
    class_indices = data["class_indices"]  # column j of `scores` corresponds to class class_indices[j]

    with pred_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    true_labels = np.array([int(r["true_label"]) for r in rows])
    pred_labels = np.array([int(r["pred_label"]) for r in rows])

    if len(true_labels) != scores.shape[0]:
        raise ValueError(
            f"predictions.csv has {len(true_labels)} rows but scores.npz has "
            f"{scores.shape[0]} -- these files are out of sync, re-run predict.py."
        )

    top1_acc = float((pred_labels == true_labels).mean())

    # top_k_accuracy_score needs the full label set via `labels=` when not
    # every class appears in y_true (common with 500 classes / 5000 test images).
    top5_acc = float(top_k_accuracy_score(true_labels, scores, k=5, labels=class_indices))

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=class_indices, average="macro", zero_division=0
    )

    return {
        "num_test_images": int(len(true_labels)),
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["frozen", "finetuned", "layer4"], default=None)
    parser.add_argument("--output_root", default="outputs/transfer")
    parser.add_argument("--output_dir", default=None,
                         help="Exact directory containing predictions.csv/scores.npz -- overrides "
                              "the official outputs/transfer/<method_name>/ path. Use this for "
                              "non-official runs (e.g. the regularization ablation).")
    parser.add_argument("--label", default=None,
                         help="Label to print/save results under when --output_dir is used "
                              "(defaults to the directory name).")
    parser.add_argument("--compare", action="store_true",
                         help="Compute metrics for both official methods (frozen + finetuned) "
                              "and print/save a side-by-side comparison table.")
    args = parser.parse_args()

    if args.output_dir is not None:
        # Single custom-directory run -- bypasses the official method_name mapping entirely.
        output_dir = Path(args.output_dir)
        label = args.label or output_dir.name
        metrics = compute_metrics_for_method(output_dir)

        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n=== {label} ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return

    if not args.compare and args.strategy is None:
        parser.error("Provide --strategy X, --output_dir X, or --compare")

    strategies = ["frozen", "finetuned"] if args.compare else [args.strategy]
    results = {}

    for strategy in strategies:
        if strategy not in STRATEGY_TO_METHOD_NAME:
            print(f"Skipping '{strategy}' -- not an official method_name.")
            continue
        method_name = STRATEGY_TO_METHOD_NAME[strategy]
        output_dir = Path(args.output_root) / method_name

        metrics = compute_metrics_for_method(output_dir)
        results[method_name] = metrics

        # Save alongside predictions.csv/scores.npz for this method
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n=== {method_name} ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if args.compare and len(results) == 2:
        summary_path = Path(args.output_root) / "metrics_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["method_name", "num_test_images", "top1_accuracy", "top5_accuracy",
                              "macro_precision", "macro_recall", "macro_f1"])
            for method_name, m in results.items():
                writer.writerow([method_name, m["num_test_images"], m["top1_accuracy"],
                                  m["top5_accuracy"], m["macro_precision"],
                                  m["macro_recall"], m["macro_f1"]])
        print(f"\nWrote comparison table to {summary_path}")


if __name__ == "__main__":
    main()
