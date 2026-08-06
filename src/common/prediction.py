"""Shared neural-model inference and artifact-writing helpers."""

from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

from src.common.runtime import unpack_batch

PREDICTION_FIELDS = (
    "image_id",
    "true_label",
    "pred_label",
    "top1_score",
    "method_name",
    "split",
)


@dataclass(frozen=True)
class PredictionResult:
    """In-memory representation of one prediction run."""

    rows: list[dict[str, Any]]
    image_ids: np.ndarray
    scores: np.ndarray
    class_indices: np.ndarray
    inference_time_seconds: float
    method_name: str

    @property
    def accuracy(self) -> float:
        predicted = np.asarray([row["pred_label"] for row in self.rows])
        expected = np.asarray([row["true_label"] for row in self.rows])
        return float(np.mean(predicted == expected))


def _to_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        converted = values.tolist()
        return converted if isinstance(converted, list) else [converted]
    return list(values)


def run_torch_prediction(
    model: torch.nn.Module,
    loader: Iterable[Any],
    *,
    device: torch.device,
    method_name: str,
    split: str = "test",
    max_batches: int | None = None,
) -> PredictionResult:
    """Run softmax inference and return artifacts in the shared output format."""

    rows: list[dict[str, Any]] = []
    all_image_ids: list[Any] = []
    all_scores: list[np.ndarray] = []
    start_time = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="predict"), start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            images, labels, image_ids = unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            probabilities = torch.softmax(model(images), dim=1)
            top1_scores, predicted_labels = probabilities.max(dim=1)

            label_values = _to_list(labels.cpu())
            prediction_values = _to_list(predicted_labels.cpu())
            top1_values = _to_list(top1_scores.cpu())
            image_id_values = _to_list(image_ids)
            if not (
                len(image_id_values)
                == len(label_values)
                == len(prediction_values)
                == len(top1_values)
            ):
                raise ValueError("Prediction batch fields have inconsistent lengths")

            rows.extend(
                {
                    "image_id": image_id,
                    "true_label": int(true_label),
                    "pred_label": int(predicted_label),
                    "top1_score": float(top1_score),
                    "method_name": method_name,
                    "split": split,
                }
                for image_id, true_label, predicted_label, top1_score in zip(
                    image_id_values,
                    label_values,
                    prediction_values,
                    top1_values,
                )
            )
            all_image_ids.extend(image_id_values)
            all_scores.append(probabilities.cpu().numpy().astype(np.float32))

    if not rows:
        raise RuntimeError("No predictions were produced")

    scores = np.concatenate(all_scores, axis=0)
    return PredictionResult(
        rows=rows,
        image_ids=np.asarray(all_image_ids),
        scores=scores,
        class_indices=np.arange(scores.shape[1], dtype=np.int64),
        inference_time_seconds=time.perf_counter() - start_time,
        method_name=method_name,
    )


def _load_runtime(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime metadata must be a JSON object: {path}")
    return payload


def write_prediction_artifacts(
    output_dir: Path,
    result: PredictionResult,
    *,
    hardware: dict[str, Any],
    software: dict[str, str],
    prediction_config: dict[str, Any],
    degradation: str | None = None,
    severity: int | None = None,
) -> None:
    """Write predictions, scores, and merged runtime metadata.

    Existing training metadata is preserved when inference writes into a model
    directory. Inference-owned fields are refreshed on every run.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(result.rows)

    np.savez(
        output_dir / "scores.npz",
        image_ids=result.image_ids,
        scores=result.scores,
        class_indices=result.class_indices,
    )

    runtime_path = output_dir / "runtime.json"
    runtime = _load_runtime(runtime_path)
    runtime.setdefault("hardware", hardware)
    runtime.setdefault("software", software)
    runtime.update(
        {
            "method_name": result.method_name,
            "inference_time_seconds": result.inference_time_seconds,
            "num_test_images": len(result.rows),
            "prediction_config": prediction_config,
            "scores_type": "softmax_probability",
            "degradation": degradation,
            "severity": severity,
            "python": platform.python_version(),
        }
    )
    temporary_path = runtime_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(runtime, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(runtime_path)
