"""Build the final cross-method tables and figures from raw model outputs.

The manifest-driven workflow keeps the submitted model artifacts outside Git
while making E's evaluation reproducible.  Every run is revalidated through
the shared output contract, recomputed with the same metric implementation,
and checked against one canonical ``image_id -> true_label`` test mapping.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.contracts import load_evaluation_data, load_runtime
from src.evaluation.metrics import calculate_metrics


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    display_name: str
    dataset_group: str
    tables: tuple[str, ...]
    predictions: Path
    scores: Path
    runtime: Path
    artifact_root: Path
    training_history: Path | None = None
    expected_method_name: str | None = None
    num_train_images: int | None = None


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_manifest(path: str | Path, artifact_root: str | Path | None = None) -> list[RunSpec]:
    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("runs"), list):
        raise ValueError(f"{manifest_path} must contain a top-level runs list")

    if artifact_root is None:
        configured_root = raw.get("artifact_root", ".")
        root = _resolve(manifest_path.parent, configured_root)
    else:
        root = Path(artifact_root).expanduser()

    specs: list[RunSpec] = []
    seen_ids: set[str] = set()
    for item in raw["runs"]:
        if not isinstance(item, dict):
            raise ValueError("each manifest run must be a mapping")
        run_id = str(item["id"])
        if run_id in seen_ids:
            raise ValueError(f"duplicate run id: {run_id}")
        seen_ids.add(run_id)
        history = item.get("training_history")
        specs.append(
            RunSpec(
                run_id=run_id,
                display_name=str(item.get("display_name", run_id)),
                dataset_group=str(item["dataset_group"]),
                tables=tuple(str(value) for value in item.get("tables", [])),
                predictions=_resolve(root, item["predictions"]),
                scores=_resolve(root, item["scores"]),
                runtime=_resolve(root, item["runtime"]),
                artifact_root=root,
                training_history=_resolve(root, history) if history else None,
                expected_method_name=(
                    str(item["expected_method_name"])
                    if item.get("expected_method_name") is not None
                    else None
                ),
                num_train_images=(
                    int(item["num_train_images"])
                    if item.get("num_train_images") is not None
                    else None
                ),
            )
        )
    if not specs:
        raise ValueError("the manifest contains no runs")
    return specs


def _hardware_label(runtime: dict[str, Any]) -> str:
    hardware = runtime.get("hardware") or {}
    gpu = hardware.get("gpu")
    if gpu:
        return str(gpu)
    platform = str(hardware.get("platform") or "unspecified")
    if platform.startswith("macOS"):
        return "macOS CPU"
    return str(hardware.get("cpu") or platform)


def _canonical_labels(data: Any) -> dict[str, int]:
    return dict(zip(data.image_ids.tolist(), data.y_true.tolist(), strict=True))


def _portable_artifact_path(path: Path, root: Path) -> str:
    """Return a shareable manifest-relative path without leaking local paths."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def evaluate_runs(
    specs: list[RunSpec], *, expected_num_classes: int = 500
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    rows: list[dict[str, Any]] = []
    evidence: dict[str, dict[str, int]] = {}
    reference_labels: dict[str, int] | None = None
    reference_id: str | None = None

    for spec in specs:
        missing = [
            path
            for path in (spec.predictions, spec.scores, spec.runtime)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{spec.run_id} is missing required artifacts: "
                + ", ".join(str(path) for path in missing)
            )
        data = load_evaluation_data(
            spec.predictions,
            spec.scores,
            expected_num_classes=expected_num_classes,
            expected_split="test",
            strict_argmax=True,
        )
        if spec.expected_method_name and data.method_name != spec.expected_method_name:
            raise ValueError(
                f"{spec.run_id}: expected method_name {spec.expected_method_name!r}, "
                f"found {data.method_name!r}"
            )

        labels = _canonical_labels(data)
        if reference_labels is None:
            reference_labels = labels
            reference_id = spec.run_id
        elif labels != reference_labels:
            reference_keys = set(reference_labels)
            keys = set(labels)
            if keys != reference_keys:
                raise ValueError(
                    f"{spec.run_id} and {reference_id} use different test image_id sets"
                )
            mismatches = [key for key in keys if labels[key] != reference_labels[key]]
            raise ValueError(
                f"{spec.run_id} and {reference_id} disagree on true_label for "
                f"{len(mismatches)} test images"
            )

        metrics = calculate_metrics(data).summary
        runtime = load_runtime(spec.runtime, len(data.predictions))
        if runtime is None:
            raise ValueError(f"runtime unexpectedly absent for {spec.run_id}")
        train_images = spec.num_train_images
        if train_images is None and runtime.get("num_train_images") is not None:
            train_images = int(runtime["num_train_images"])

        row = {
            "run_id": spec.run_id,
            "display_name": spec.display_name,
            "method_name": data.method_name,
            "dataset_group": spec.dataset_group,
            "tables": ",".join(spec.tables),
            "num_train_images": train_images,
            "num_test_images": len(data.predictions),
            **metrics,
            "training_time_seconds": float(runtime.get("training_time_seconds", np.nan)),
            "inference_time_seconds": float(runtime["inference_time_seconds"]),
            "best_epoch": runtime.get("best_epoch"),
            "best_val_accuracy": runtime.get("best_val_accuracy"),
            "hardware": _hardware_label(runtime),
            "predictions": _portable_artifact_path(spec.predictions, spec.artifact_root),
            "scores": _portable_artifact_path(spec.scores, spec.artifact_root),
            "runtime": _portable_artifact_path(spec.runtime, spec.artifact_root),
        }
        rows.append(row)
        evidence[spec.run_id] = {
            "num_test_images": int(len(data.predictions)),
            "num_classes": int(len(data.labels)),
        }

    return pd.DataFrame(rows), evidence


METRIC_COLUMNS = ["top1_accuracy", "top5_accuracy", "macro_f1", "balanced_accuracy"]


def _latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _metric_ranks(table: pd.DataFrame, column: str) -> dict[int, int]:
    values = table[column].dropna().unique()
    ordered = sorted((float(value) for value in values), reverse=True)
    return {index: ordered.index(float(value)) + 1 for index, value in table[column].items()}


def _format_metric(value: float, rank: int) -> str:
    rendered = f"{100.0 * float(value):.2f}"
    if rank == 1:
        return rf"\textbf{{{rendered}}}"
    if rank == 2:
        return rf"\underline{{{rendered}}}"
    return rendered


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    ranks = {column: _metric_ranks(table, column) for column in METRIC_COLUMNS}
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Top-1 & Top-5 & Macro-F1 & Bal. Acc. & Train (s) & Infer (s) \\",
        r"\midrule",
    ]
    for index, row in table.iterrows():
        metrics = [
            _format_metric(row[column], ranks[column][index]) for column in METRIC_COLUMNS
        ]
        train_time = (
            "--"
            if pd.isna(row["training_time_seconds"])
            else f"{float(row['training_time_seconds']):.1f}"
        )
        inference_time = f"{float(row['inference_time_seconds']):.1f}"
        lines.append(
            " & ".join(
                [_latex_escape(str(row["display_name"])), *metrics, train_time, inference_time]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_performance_time(table: pd.DataFrame, output_dir: Path) -> None:
    if table.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    hardware_values = list(table["hardware"].drop_duplicates())
    palette = plt.get_cmap("tab10")
    colours = {name: palette(i % 10) for i, name in enumerate(hardware_values)}
    panels = [
        ("training_time_seconds", "Training time (seconds, log scale)"),
        ("inference_time_seconds", "Inference time for 5,000 images (seconds)"),
    ]
    for axis, (column, xlabel) in zip(axes, panels, strict=True):
        for _, row in table.iterrows():
            x = float(row[column])
            axis.scatter(
                x,
                100.0 * float(row["top1_accuracy"]),
                s=72,
                color=colours[row["hardware"]],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            axis.annotate(
                str(row["display_name"]),
                (x, 100.0 * float(row["top1_accuracy"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )
        if column == "training_time_seconds":
            axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test Top-1 accuracy (%)")
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor=colour,
            markeredgecolor="black", label=hardware, markersize=7
        )
        for hardware, colour in colours.items()
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=min(3, len(handles)),
        fontsize=8,
    )
    figure.suptitle("Limited-data performance and recorded runtime")
    figure.text(
        0.5,
        0.018,
        "Runtime points were recorded on different hardware and are descriptive, not controlled speed comparisons.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.17, 1, 0.96))
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"main_performance_vs_time.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_full_training_curves(specs: list[RunSpec], output_dir: Path) -> None:
    selected = [spec for spec in specs if "full_data" in spec.tables and spec.training_history]
    if not selected:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for spec in selected:
        if spec.training_history is None or not spec.training_history.is_file():
            raise FileNotFoundError(f"missing training history for {spec.run_id}")
        history = pd.read_csv(spec.training_history)
        axes[0].plot(history["epoch"], 100.0 * history["train_acc"], label=f"{spec.display_name} train")
        axes[0].plot(history["epoch"], 100.0 * history["val_acc"], linestyle="--", label=f"{spec.display_name} val")
        axes[1].plot(history["epoch"], history["train_loss"], label=f"{spec.display_name} train")
        axes[1].plot(history["epoch"], history["val_loss"], linestyle="--", label=f"{spec.display_name} val")
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel("Cross-entropy loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(range(1, 16, 2))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    axes[0].set_title("Full-data accuracy")
    axes[1].set_title("Full-data loss")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"full_data_training_curves.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_report_artifacts(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    artifact_root: str | Path | None = None,
    expected_num_classes: int = 500,
) -> pd.DataFrame:
    specs = load_manifest(manifest_path, artifact_root=artifact_root)
    summary, evidence = evaluate_runs(specs, expected_num_classes=expected_num_classes)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "all_results.csv", index=False)
    (output / "evaluation_evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    table_names = sorted({name for spec in specs for name in spec.tables})
    for table_name in table_names:
        ids = [spec.run_id for spec in specs if table_name in spec.tables]
        table = summary.set_index("run_id").loc[ids].reset_index()
        table.to_csv(output / f"{table_name}.csv", index=False)
        write_latex_table(table, output / f"{table_name}.tex")

    main = summary.loc[summary["run_id"].isin(
        [spec.run_id for spec in specs if "main" in spec.tables]
    )]
    _plot_performance_time(main, output)
    _plot_full_training_curves(specs, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-num-classes", type=int, default=500)
    args = parser.parse_args()
    summary = build_report_artifacts(
        args.manifest,
        args.output,
        artifact_root=args.artifact_root,
        expected_num_classes=args.expected_num_classes,
    )
    print(
        f"Validated {len(summary)} runs on one shared test mapping; "
        f"report artifacts written to {args.output}"
    )


if __name__ == "__main__":
    main()
