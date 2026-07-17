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


def run_robustness_inference(
    predictor: ModelPredictor,
    samples: list[RobustnessSample],
    output_root: str | Path,
    *,
    config_path: str | Path = "configs/robustness.yaml",
    base_seed: int = 9517,
    hardware: dict[str, object] | None = None,
    software: dict[str, object] | None = None,
) -> list[Path]:
    """Run the complete degradation matrix through a model-like predictor.

    This is deliberately a small, model-agnostic MVP interface. Real model
    code only needs a thin adapter implementing ``ModelPredictor``.
    """

    if not samples:
        raise ValueError("at least one robustness sample is required")
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
            started = time.perf_counter()
            score_rows: list[np.ndarray] = []
            prediction_rows: list[dict[str, object]] = []
            for sample in samples:
                with Image.open(sample.image_path) as source:
                    rgb = source.convert("RGB")
                degraded = apply_degradation(
                    rgb,
                    degradation_type,
                    int(severity),
                    seed=degradation_seed(sample.image_id, base_seed),
                )
                scores = np.asarray(predictor.predict_scores(degraded), dtype=float)
                if scores.shape != (len(class_indices),):
                    raise ValueError(
                        f"predict_scores must return shape {(len(class_indices),)}; got {scores.shape}"
                    )
                if not np.isfinite(scores).all():
                    raise ValueError("predict_scores returned NaN or infinity")
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
            run_dir = (
                output_root
                / predictor.method_name
                / degradation_type
                / f"severity_{severity}"
            )
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

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)
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
    args = parser.parse_args()
    summary = evaluate_robustness(
        args.root,
        args.output,
        methods=args.methods,
        expected_num_classes=args.expected_num_classes,
        config_path=args.config,
        allow_missing=args.allow_missing,
    )
    print(
        f"Aggregated {len(summary)} robustness runs; outputs written to {args.output}"
    )


if __name__ == "__main__":
    main()
