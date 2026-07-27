#!/usr/bin/env python3
"""Run the final 4x5 robustness matrix for the selected real models."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.model_adapters import (  # noqa: E402
    HOGRandomForestPredictor,
    SIFTBoVWSVMPredictor,
    TorchResNetPredictor,
)
from src.evaluation.robustness import (  # noqa: E402
    RobustnessSample,
    evaluate_robustness,
    run_clean_inference,
    run_robustness_inference,
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_samples(test_csv: Path, data_root: Path) -> list[RobustnessSample]:
    frame = pd.read_csv(test_csv, dtype={"image_id": "string"})
    required = {"image_id", "image_path", "class_idx", "split"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{test_csv} is missing columns: {missing_columns}")
    if set(frame["split"].astype(str)) != {"test"}:
        raise ValueError(f"{test_csv} must contain only split=test rows")
    samples = [
        RobustnessSample(
            image_id=str(row.image_id),
            true_label=int(row.class_idx),
            image_path=data_root / str(row.image_path),
        )
        for row in frame.itertuples(index=False)
    ]
    missing = [sample.image_path for sample in samples if not sample.image_path.is_file()]
    if missing:
        preview = "\n- ".join(str(path) for path in missing[:8])
        raise FileNotFoundError(
            f"{len(missing)} of {len(samples)} test images are missing under {data_root}:\n- {preview}"
        )
    return samples


def load_predictor(spec: dict[str, Any], artifact_root: Path, device: str) -> Any:
    kind = str(spec["kind"])
    if kind == "hog_random_forest":
        return HOGRandomForestPredictor.from_pickle(
            _resolve(artifact_root, spec["model"]),
            image_size=int(spec.get("image_size", 128)),
            pixels_per_cell=int(spec.get("pixels_per_cell", 16)),
        )
    if kind == "sift_bovw_svm":
        return SIFTBoVWSVMPredictor.from_pickles(
            _resolve(artifact_root, spec["vocabulary"]),
            _resolve(artifact_root, spec["model"]),
            max_desc_per_image=int(spec.get("max_desc_per_image", 200)),
            feature_seed=int(spec.get("feature_seed", 9517)),
        )
    if kind == "scratch_resnet18":
        return TorchResNetPredictor.from_scratch_checkpoint(
            _resolve(artifact_root, spec["checkpoint"]),
            method_name=str(spec["method_name"]),
            device=device,
            image_size=int(spec.get("image_size", 224)),
        )
    if kind == "transfer_resnet18":
        return TorchResNetPredictor.from_transfer_checkpoint(
            _resolve(artifact_root, spec["checkpoint"]),
            method_name=str(spec["method_name"]),
            device=device,
            image_size=int(spec.get("image_size", 224)),
            use_attention=bool(spec.get("use_attention", False)),
            num_heads=int(spec.get("num_heads", 8)),
            dropout_rate=float(spec.get("dropout_rate", 0.0)),
        )
    raise ValueError(f"unknown model kind: {kind}")


def verify_clean_predictions(
    local_csv: Path,
    reference_csv: Path,
    *,
    num_images: int,
    minimum_agreement: float,
    max_top1_delta: float,
) -> None:
    """Check local clean inference against a submitted prediction artifact."""

    if num_images <= 0:
        return
    local = pd.read_csv(local_csv, dtype={"image_id": "string"})
    reference = pd.read_csv(reference_csv, dtype={"image_id": "string"})
    required = {"image_id", "true_label", "pred_label"}
    for name, frame, path in (
        ("local", local, local_csv),
        ("reference", reference, reference_csv),
    ):
        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{name} clean file {path} is missing: {missing_columns}")
        if frame["image_id"].duplicated().any():
            raise ValueError(f"{name} clean file contains duplicate image IDs: {path}")

    selected = local.iloc[: min(num_images, len(local))].copy()
    aligned_reference = reference.set_index("image_id").reindex(selected["image_id"])
    if aligned_reference["pred_label"].isna().any():
        absent = selected.loc[
            aligned_reference["pred_label"].isna().to_numpy(), "image_id"
        ].tolist()
        raise ValueError(f"clean reference does not contain test IDs: {absent[:8]}")
    if not np.array_equal(
        selected["true_label"].to_numpy(dtype=int),
        aligned_reference["true_label"].to_numpy(dtype=int),
    ):
        raise ValueError("local and submitted clean files disagree on true labels")

    local_pred = selected["pred_label"].to_numpy(dtype=int)
    reference_pred = aligned_reference["pred_label"].to_numpy(dtype=int)
    true_labels = selected["true_label"].to_numpy(dtype=int)
    agreement = float(np.mean(local_pred == reference_pred))
    local_top1 = float(np.mean(local_pred == true_labels))
    reference_top1 = float(np.mean(reference_pred == true_labels))
    top1_delta = abs(local_top1 - reference_top1)
    if agreement < minimum_agreement or top1_delta > max_top1_delta:
        raise ValueError(
            f"clean preflight failed for {local_csv}: agreement={agreement:.4f} "
            f"(minimum {minimum_agreement:.4f}), Top-1 delta={top1_delta:.4f} "
            f"(maximum {max_top1_delta:.4f})"
        )
    print(
        f"Verified {len(selected)} clean predictions: agreement={agreement:.4f}, "
        f"local Top-1={local_top1:.4f}, submitted Top-1={reference_top1:.4f}."
    )


def software_info() -> dict[str, str]:
    names = ["numpy", "scikit-learn", "opencv-python", "scikit-image", "torch", "torchvision"]
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("configs/final_robustness.yaml"))
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--evaluation-output", type=Path, default=Path("outputs/evaluation/robustness"))
    parser.add_argument("--config", type=Path, default=Path("configs/robustness.yaml"))
    parser.add_argument(
        "--clean-results",
        type=Path,
        default=None,
        help="Optional external clean summary; by default local severity-0 runs are used.",
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--clean-check-images",
        type=int,
        default=5000,
        help="Number of local clean predictions compared with submitted artifacts.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load models and reproduce clean predictions without running degradations.",
    )
    args = parser.parse_args()

    model_config = yaml.safe_load(args.models.read_text(encoding="utf-8"))
    model_specs = model_config.get("models", [])
    if args.methods:
        selected = set(args.methods)
        model_specs = [spec for spec in model_specs if str(spec["method_name"]) in selected]
        absent = selected - {str(spec["method_name"]) for spec in model_specs}
        if absent:
            raise ValueError(f"unknown requested methods: {sorted(absent)}")
    if not model_specs:
        raise ValueError("no robustness models selected")

    samples = load_samples(args.test_csv, args.data_root)
    hardware = {
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "Apple MPS"
            if torch.backends.mps.is_available()
            else None
        ),
    }
    software = software_info()
    completed_methods: list[str] = []
    for spec in model_specs:
        method_name = str(spec["method_name"])
        print(f"Loading {method_name}...")
        predictor = load_predictor(spec, args.artifact_root, args.device)
        if "clean_predictions" not in spec:
            raise ValueError(f"{method_name} is missing clean_predictions in {args.models}")
        clean_dir = run_clean_inference(
            predictor,
            samples,
            args.output,
            hardware=hardware,
            software=software,
            batch_size=args.batch_size,
            skip_existing=args.resume,
        )
        verify_clean_predictions(
            clean_dir / "predictions.csv",
            _resolve(args.artifact_root, spec["clean_predictions"]),
            num_images=args.clean_check_images,
            minimum_agreement=float(spec.get("minimum_clean_agreement", 0.999)),
            max_top1_delta=float(spec.get("max_clean_top1_delta", 0.001)),
        )
        if args.preflight_only:
            completed_methods.append(method_name)
            del predictor
            continue
        run_robustness_inference(
            predictor,
            samples,
            args.output,
            config_path=args.config,
            hardware=hardware,
            software=software,
            batch_size=args.batch_size,
            skip_existing=args.resume,
        )
        completed_methods.append(method_name)
        del predictor

    if args.preflight_only:
        print(f"Clean-prediction preflight passed for {len(completed_methods)} models")
        return

    summary = evaluate_robustness(
        args.output,
        args.evaluation_output,
        methods=completed_methods,
        config_path=args.config,
        clean_results=args.clean_results,
    )
    (args.evaluation_output / "run_manifest.json").write_text(
        json.dumps(
            {
                "methods": completed_methods,
                "num_test_images": len(samples),
                "test_csv": str(args.test_csv),
                "data_root": str(args.data_root),
                "artifact_root": str(args.artifact_root),
                "clean_check_images": min(args.clean_check_images, len(samples)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Completed {len(summary)} robustness rows for {len(completed_methods)} models")


if __name__ == "__main__":
    main()
