from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import yaml
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.contracts import load_evaluation_data, load_runtime
from src.evaluation.degradation import apply_degradation, degradation_seed
from src.evaluation.metrics import calculate_metrics
from src.evaluation.plots import plot_robustness_curves


@dataclass(frozen=True)
class RobustnessSample:
    """One labelled RGB image made available to the shared robustness runner."""

    image_id: str
    true_label: int
    image_path: Path


class ModelPredictor(Protocol):
    """Minimal model interface required by the shared robustness runner.

    Each model adapter keeps its own resize, feature extraction and
    normalisation inside ``predict_scores``. The runner always applies image
    degradation first, on the raw RGB PIL image.
    """

    method_name: str
    class_indices: np.ndarray

    def predict_scores(self, image: Image.Image) -> np.ndarray:
        """Return one score per entry in ``class_indices`` for a single image."""
        ...


def run_clean_inference(
    predictor: ModelPredictor,
    samples: list[RobustnessSample],
    output_root: str | Path,
    *,
    hardware: dict[str, object] | None = None,
    software: dict[str, object] | None = None,
    batch_size: int = 64,
    skip_existing: bool = False,
) -> Path:
    """Run one undegraded pass used as the same-environment severity-0 baseline."""

    if not samples:
        raise ValueError("at least one clean sample is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    class_indices = np.asarray(predictor.class_indices)
    if class_indices.ndim != 1 or not np.issubdtype(class_indices.dtype, np.integer):
        raise ValueError("predictor.class_indices must be a 1-D integer array")
    class_indices = class_indices.astype(np.int64)
    if len(np.unique(class_indices)) != len(class_indices):
        raise ValueError("predictor.class_indices must not contain duplicates")
    sample_ids = [str(sample.image_id) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("clean sample image_id values must be unique")

    run_dir = Path(output_root) / predictor.method_name / "clean" / "severity_0"
    required_outputs = [
        run_dir / "predictions.csv",
        run_dir / "scores.npz",
        run_dir / "runtime.json",
    ]
    if skip_existing and all(path.is_file() for path in required_outputs):
        return run_dir

    begin_run = getattr(predictor, "begin_run", None)
    if callable(begin_run):
        begin_run("clean", 0)
    started = time.perf_counter()
    score_rows: list[np.ndarray] = []
    prediction_rows: list[dict[str, object]] = []
    for offset in range(0, len(samples), batch_size):
        sample_batch = samples[offset : offset + batch_size]
        images: list[Image.Image] = []
        image_ids: list[str] = []
        for sample in sample_batch:
            if not Path(sample.image_path).is_file():
                raise FileNotFoundError(sample.image_path)
            with Image.open(sample.image_path) as source:
                images.append(source.convert("RGB"))
            image_ids.append(str(sample.image_id))

        batch_predictor = getattr(predictor, "predict_scores_batch", None)
        if callable(batch_predictor):
            batch_scores = np.asarray(
                batch_predictor(images, image_ids), dtype=float
            )
        else:
            batch_scores = np.vstack(
                [predictor.predict_scores(image) for image in images]
            ).astype(float)
        expected_shape = (len(sample_batch), len(class_indices))
        if batch_scores.shape != expected_shape:
            raise ValueError(
                f"predictor must return clean batch shape {expected_shape}; "
                f"got {batch_scores.shape}"
            )
        if not np.isfinite(batch_scores).all():
            raise ValueError("predictor returned NaN or infinity on clean images")
        for sample, scores in zip(sample_batch, batch_scores, strict=True):
            pred_label = int(class_indices[int(np.argmax(scores))])
            score_rows.append(scores)
            prediction_rows.append(
                {
                    "image_id": str(sample.image_id),
                    "true_label": int(sample.true_label),
                    "pred_label": pred_label,
                    "top1_score": float(np.max(scores)),
                    "method_name": predictor.method_name,
                    "split": "test",
                }
            )

    elapsed = time.perf_counter() - started
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False)
    np.savez(
        run_dir / "scores.npz",
        image_ids=np.asarray(sample_ids),
        scores=np.vstack(score_rows),
        class_indices=class_indices,
    )
    runtime = {
        "inference_time_seconds": elapsed,
        "num_test_images": len(samples),
        "hardware": hardware or {"platform": "unspecified", "cpu": None, "gpu": None},
        "software": software
        or {"python": f"{sys.version_info.major}.{sys.version_info.minor}"},
    }
    (run_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_dir


def run_robustness_inference(
    predictor: ModelPredictor,
    samples: list[RobustnessSample],
    output_root: str | Path,
    *,
    config_path: str | Path = "configs/robustness.yaml",
    base_seed: int = 9517,
    hardware: dict[str, object] | None = None,
    software: dict[str, object] | None = None,
    batch_size: int = 64,
    skip_existing: bool = False,
) -> list[Path]:
    """Run the complete degradation matrix through a model-like predictor.

    This is deliberately a small, model-agnostic MVP interface. Real model
    code only needs a thin adapter implementing ``ModelPredictor``.
    """

    if not samples:
        raise ValueError("at least one robustness sample is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    class_indices = np.asarray(predictor.class_indices)
    if class_indices.ndim != 1 or not np.issubdtype(class_indices.dtype, np.integer):
        raise ValueError("predictor.class_indices must be a 1-D integer array")
    class_indices = class_indices.astype(np.int64)
    if len(np.unique(class_indices)) != len(class_indices):
        raise ValueError("predictor.class_indices must not contain duplicates")
    if not predictor.method_name:
        raise ValueError("predictor.method_name cannot be empty")

    sample_ids = [str(sample.image_id) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("robustness sample image_id values must be unique")
    for sample in samples:
        if not Path(sample.image_path).is_file():
            raise FileNotFoundError(sample.image_path)
        if int(sample.true_label) not in set(class_indices.tolist()):
            raise ValueError(
                f"true_label {sample.true_label} is absent from predictor.class_indices"
            )

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output_root = Path(output_root)
    written_run_dirs: list[Path] = []
    for degradation_type in config["degradations"]:
        for severity in config["severity_levels"]:
            run_dir = (
                output_root
                / predictor.method_name
                / degradation_type
                / f"severity_{severity}"
            )
            required_outputs = [
                run_dir / "predictions.csv",
                run_dir / "scores.npz",
                run_dir / "runtime.json",
            ]
            if skip_existing and all(path.is_file() for path in required_outputs):
                written_run_dirs.append(run_dir)
                continue

            begin_run = getattr(predictor, "begin_run", None)
            if callable(begin_run):
                begin_run(degradation_type, int(severity))

            started = time.perf_counter()
            score_rows: list[np.ndarray] = []
            prediction_rows: list[dict[str, object]] = []
            for offset in range(0, len(samples), batch_size):
                sample_batch = samples[offset : offset + batch_size]
                degraded_images: list[Image.Image] = []
                image_ids: list[str] = []
                for sample in sample_batch:
                    with Image.open(sample.image_path) as source:
                        rgb = source.convert("RGB")
                    degraded_images.append(
                        apply_degradation(
                            rgb,
                            degradation_type,
                            int(severity),
                            seed=degradation_seed(sample.image_id, base_seed),
                        )
                    )
                    image_ids.append(str(sample.image_id))

                batch_predictor = getattr(predictor, "predict_scores_batch", None)
                if callable(batch_predictor):
                    batch_scores = np.asarray(
                        batch_predictor(degraded_images, image_ids), dtype=float
                    )
                else:
                    batch_scores = np.vstack(
                        [predictor.predict_scores(image) for image in degraded_images]
                    ).astype(float)
                expected_shape = (len(sample_batch), len(class_indices))
                if batch_scores.shape != expected_shape:
                    raise ValueError(
                        f"predictor must return batch shape {expected_shape}; "
                        f"got {batch_scores.shape}"
                    )
                if not np.isfinite(batch_scores).all():
                    raise ValueError("predictor returned NaN or infinity")

                for sample, scores in zip(sample_batch, batch_scores, strict=True):
                    pred_label = int(class_indices[int(np.argmax(scores))])
                    score_rows.append(scores)
                    prediction_rows.append(
                        {
                            "image_id": str(sample.image_id),
                            "true_label": int(sample.true_label),
                            "pred_label": pred_label,
                            "top1_score": float(np.max(scores)),
                            "method_name": predictor.method_name,
                            "split": "test",
                        }
                    )

            elapsed = time.perf_counter() - started
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(prediction_rows).to_csv(
                run_dir / "predictions.csv", index=False
            )
            np.savez(
                run_dir / "scores.npz",
                image_ids=np.asarray(sample_ids),
                scores=np.vstack(score_rows),
                class_indices=class_indices,
            )
            runtime = {
                "inference_time_seconds": elapsed,
                "num_test_images": len(samples),
                "hardware": hardware
                or {"platform": "unspecified", "cpu": None, "gpu": None},
                "software": software
                or {"python": f"{sys.version_info.major}.{sys.version_info.minor}"},
            }
            (run_dir / "runtime.json").write_text(
                json.dumps(runtime, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written_run_dirs.append(run_dir)
    return written_run_dirs


def evaluate_robustness(
    robustness_root: str | Path,
    output_dir: str | Path,
    *,
    methods: list[str] | None = None,
    expected_num_classes: int | None = 500,
    config_path: str | Path = "configs/robustness.yaml",
    allow_missing: bool = False,
    clean_results: str | Path | None = None,
) -> pd.DataFrame:
    robustness_root = Path(robustness_root)
    output_dir = Path(output_dir)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if methods is None:
        methods = sorted(
            path.name for path in robustness_root.iterdir() if path.is_dir()
        )
    if not methods:
        raise ValueError("no robustness methods were supplied or discovered")

    rows: list[dict[str, object]] = []
    missing_runs: list[str] = []
    levels = config["severity_levels"]
    for method_name in methods:
        for degradation_type, degradation_config in config["degradations"].items():
            values = degradation_config["values"]
            for severity, parameter_value in zip(levels, values, strict=True):
                run_dir = (
                    robustness_root
                    / method_name
                    / degradation_type
                    / f"severity_{severity}"
                )
                predictions_path = run_dir / "predictions.csv"
                scores_path = run_dir / "scores.npz"
                runtime_path = run_dir / "runtime.json"
                required = [predictions_path, scores_path, runtime_path]
                absent = [str(path) for path in required if not path.exists()]
                if absent:
                    missing_runs.extend(absent)
                    continue

                data = load_evaluation_data(
                    predictions_path,
                    scores_path,
                    expected_num_classes=expected_num_classes,
                    expected_split="test",
                    strict_argmax=True,
                )
                if data.method_name != method_name:
                    raise ValueError(
                        f"folder method {method_name!r} disagrees with predictions method_name {data.method_name!r}"
                    )
                metrics = calculate_metrics(data).summary
                runtime = load_runtime(runtime_path, len(data.predictions)) or {}
                rows.append(
                    {
                        "method_name": method_name,
                        "degradation_type": degradation_type,
                        "severity": int(severity),
                        "parameter": degradation_config["parameter"],
                        "parameter_value": parameter_value,
                        "num_samples": int(len(data.predictions)),
                        **metrics,
                        "inference_time_seconds": float(
                            runtime["inference_time_seconds"]
                        ),
                        "run_dir": str(run_dir),
                    }
                )

    if missing_runs and not allow_missing:
        preview = "\n- ".join(missing_runs[:12])
        more = (
            f"\n... and {len(missing_runs) - 12} more" if len(missing_runs) > 12 else ""
        )
        raise FileNotFoundError(
            f"robustness matrix is incomplete; missing:\n- {preview}{more}"
        )
    if not rows:
        raise ValueError("no complete robustness runs were found")

    summary = pd.DataFrame(rows).sort_values(
        ["degradation_type", "method_name", "severity"]
    )
    expected_rows = len(methods) * len(config["degradations"]) * len(levels)
    if not allow_missing and len(summary) != expected_rows:
        raise AssertionError(
            f"expected {expected_rows} robustness rows, found {len(summary)}"
        )

    if clean_results is None:
        clean_rows: list[dict[str, object]] = []
        present_clean_runs = 0
        for method_name in methods:
            run_dir = robustness_root / method_name / "clean" / "severity_0"
            predictions_path = run_dir / "predictions.csv"
            scores_path = run_dir / "scores.npz"
            if predictions_path.is_file() and scores_path.is_file():
                present_clean_runs += 1
                data = load_evaluation_data(
                    predictions_path,
                    scores_path,
                    expected_num_classes=expected_num_classes,
                    expected_split="test",
                    strict_argmax=True,
                )
                metrics = calculate_metrics(data).summary
                clean_rows.append(
                    {
                        "run_id": method_name,
                        "top1_accuracy": metrics["top1_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                    }
                )
        if present_clean_runs not in {0, len(methods)}:
            raise FileNotFoundError(
                "same-environment clean baselines exist for only some robustness methods"
            )
        clean = pd.DataFrame(clean_rows) if clean_rows else None
    else:
        clean = pd.read_csv(clean_results)

    if clean is not None:
        required_clean_columns = {"run_id", "top1_accuracy", "macro_f1"}
        missing_columns = sorted(required_clean_columns - set(clean.columns))
        if missing_columns:
            raise ValueError(
                f"clean results are missing columns: {missing_columns}"
            )
        if clean["run_id"].duplicated().any():
            raise ValueError("clean results contain duplicate run_id values")
        clean = clean.set_index("run_id")
        absent_methods = sorted(set(methods) - set(clean.index))
        if absent_methods:
            raise ValueError(
                f"clean results do not cover robustness methods: {absent_methods}"
            )
        summary["clean_top1_accuracy"] = summary["method_name"].map(
            clean["top1_accuracy"]
        )
        summary["clean_macro_f1"] = summary["method_name"].map(clean["macro_f1"])
        summary["top1_drop"] = (
            summary["clean_top1_accuracy"] - summary["top1_accuracy"]
        )
        summary["macro_f1_drop"] = (
            summary["clean_macro_f1"] - summary["macro_f1"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)
    summary.drop(columns=["run_dir"]).to_csv(
        output_dir / "robustness_report.csv", index=False
    )
    plot_robustness_curves(summary, output_dir / "robustness_curves.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, aggregate, and plot all robustness runs."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/robustness.yaml"))
    parser.add_argument("--expected-num-classes", type=int, default=500)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--clean-results", type=Path, default=None)
    args = parser.parse_args()
    summary = evaluate_robustness(
        args.root,
        args.output,
        methods=args.methods,
        expected_num_classes=args.expected_num_classes,
        config_path=args.config,
        allow_missing=args.allow_missing,
        clean_results=args.clean_results,
    )
    print(
        f"Aggregated {len(summary)} robustness runs; outputs written to {args.output}"
    )


if __name__ == "__main__":
    main()
