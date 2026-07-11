from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
    output_path: str | Path,
    *,
    title: str,
    normalise: bool = False,
) -> None:
    shown = matrix.astype(float)
    colour_label = "Count"
    if normalise:
        totals = shown.sum(axis=1, keepdims=True)
        shown = np.divide(shown, totals, out=np.zeros_like(shown), where=totals != 0)
        colour_label = "Fraction of true class"

    count = len(labels)
    side = 12 if count > 30 else max(7, min(14, 0.55 * count + 4))
    figure, axis = plt.subplots(figsize=(side, side * 0.86))
    image = axis.imshow(shown, interpolation="nearest", cmap="viridis", aspect="auto")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=colour_label)
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    if count <= 30:
        tick_labels = [f"{int(label)}\n{class_names[int(label)]}" for label in labels]
        axis.set_xticks(np.arange(count), tick_labels, rotation=90, fontsize=7)
        axis.set_yticks(np.arange(count), tick_labels, fontsize=7)
        if count <= 15:
            threshold = shown.max() / 2 if shown.size else 0
            for row in range(count):
                for column in range(count):
                    value = shown[row, column]
                    text = f"{value:.2f}" if normalise else str(int(value))
                    axis.text(
                        column,
                        row,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > threshold else "black",
                    )
    else:
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix_outputs(
    matrix: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
    output_dir: str | Path,
    *,
    title: str,
) -> tuple[Path, Path]:
    """Save the full confusion matrix as both NumPy data and a PNG figure."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "confusion_matrix.npy"
    png_path = output_dir / "confusion_matrix.png"
    np.save(npy_path, matrix)
    plot_confusion_matrix(matrix, labels, class_names, png_path, title=title)
    return npy_path, png_path


def plot_confused_subset(
    matrix: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
    confused_pairs: "object",
    output_path: str | Path,
    *,
    title: str,
    max_classes: int = 20,
) -> bool:
    if getattr(confused_pairs, "empty", True):
        return False
    selected: list[int] = []
    for row in confused_pairs.itertuples(index=False):
        for label in (int(row.true_label), int(row.pred_label)):
            if label not in selected:
                selected.append(label)
            if len(selected) >= max_classes:
                break
        if len(selected) >= max_classes:
            break
    positions = [int(np.flatnonzero(labels == label)[0]) for label in selected]
    subset = matrix[np.ix_(positions, positions)]
    plot_confusion_matrix(
        subset,
        np.asarray(selected, dtype=np.int64),
        class_names,
        output_path,
        title=title,
        normalise=True,
    )
    return True
