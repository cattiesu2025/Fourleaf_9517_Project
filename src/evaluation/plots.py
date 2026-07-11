from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_robustness_curves(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Plot Top-1 accuracy and Macro F1 across degradation severity levels."""

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

    for axis, degradation in zip(axes.flat, degradations, strict=False):
        subset = summary.loc[summary["degradation_type"] == degradation]
        for method_name, method_rows in subset.groupby("method_name", sort=False):
            method_rows = method_rows.sort_values("severity")
            axis.plot(
                method_rows["severity"],
                method_rows["top1_accuracy"],
                marker="o",
                label=f"{method_name} Top-1",
            )
            axis.plot(
                method_rows["severity"],
                method_rows["macro_f1"],
                marker="s",
                linestyle="--",
                label=f"{method_name} Macro F1",
            )
        axis.set_title(degradation.replace("_", " ").title())
        axis.set_xlabel("Severity")
        axis.set_ylabel("Score")
        axis.set_xticks(sorted(subset["severity"].unique()))
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)

    for axis in axes.flat[len(degradations) :]:
        axis.set_visible(False)

    figure.suptitle("Test-time robustness", y=1.01)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
