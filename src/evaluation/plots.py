from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_robustness_curves(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Plot paper-ready Top-1 accuracy across degradation severity levels."""

    degradations = list(summary["degradation_type"].drop_duplicates())
    if not degradations:
        raise ValueError("robustness summary contains no rows")

    columns = 2
    rows = (len(degradations) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13, 4.8 * rows),
        squeeze=False,
        sharey=True,
    )
    display_names = {
        "hog_random_forest": "HOG + RF",
        "sift_bovw_svm": "SIFT + BoVW + SVM",
        "resnet18_scratch_basic_aug_sgd": "Scratch ResNet18",
        "resnet18_pretrained_finetuned": "Fine-tuned ResNet18",
    }
    degradation_titles = {
        "gaussian_noise": "Gaussian Noise",
        "blur": "Blur",
        "brightness": "Brightness",
        "jpeg_compression": "JPEG Compression",
    }

    for axis, degradation in zip(axes.flat, degradations, strict=False):
        subset = summary.loc[summary["degradation_type"] == degradation]
        for method_name, method_rows in subset.groupby("method_name", sort=False):
            method_rows = method_rows.sort_values("severity")
            severities = method_rows["severity"].tolist()
            top1_values = (100.0 * method_rows["top1_accuracy"]).tolist()
            if {"clean_top1_accuracy", "clean_macro_f1"}.issubset(method_rows.columns):
                severities = [0, *severities]
                top1_values = [
                    100.0 * float(method_rows["clean_top1_accuracy"].iloc[0]),
                    *top1_values,
                ]
            axis.plot(
                severities,
                top1_values,
                marker="o",
                linewidth=1.8,
                label=display_names.get(method_name, method_name),
            )
        axis.set_title(degradation_titles.get(degradation, degradation.replace("_", " ").title()))
        axis.set_xlabel("Severity")
        axis.set_ylabel("Top-1 accuracy (%)")
        severity_ticks = sorted(subset["severity"].unique())
        if "clean_top1_accuracy" in subset.columns:
            severity_ticks = [0, *severity_ticks]
        axis.set_xticks(severity_ticks)
        axis.set_ylim(0, 70)
        axis.grid(alpha=0.25)

    for axis in axes.flat[len(degradations) :]:
        axis.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize=9,
    )
    figure.suptitle("Robustness to test-time image degradation", y=0.98)
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
