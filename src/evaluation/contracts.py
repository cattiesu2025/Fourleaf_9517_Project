from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = (
    "image_id",
    "true_label",
    "pred_label",
    "top1_score",
    "method_name",
    "split",
)
SCORES_KEYS = ("image_ids", "scores", "class_indices")


@dataclass(frozen=True)
class EvaluationData:
    predictions: pd.DataFrame
    scores: np.ndarray
    class_indices: np.ndarray
    labels: np.ndarray
    method_name: str
    split: str

    @property
    def image_ids(self) -> np.ndarray:
        return self.predictions["image_id"].to_numpy(dtype=str)

    @property
    def y_true(self) -> np.ndarray:
        return self.predictions["true_label"].to_numpy(dtype=np.int64)

    @property
    def y_pred(self) -> np.ndarray:
        return self.predictions["pred_label"].to_numpy(dtype=np.int64)


def _normalise_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.integer):
        return str(int(value))
    return str(value)


def load_class_names(path: str | Path | None, labels: np.ndarray) -> dict[int, str]:
    names = {int(label): str(int(label)) for label in labels}
    if path is None:
        return names

    mapping_path = Path(path)
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    for label in labels:
        key = str(int(label))
        if key not in raw:
            raise ValueError(f"{mapping_path} is missing class_idx {key}")
        value = raw[key]
        if isinstance(value, dict):
            names[int(label)] = str(value.get("class_name", value.get("name", key)))
        else:
            names[int(label)] = str(value)
    return names


def load_runtime(path: str | Path | None, expected_samples: int) -> dict[str, Any] | None:
    if path is None:
        return None
    runtime_path = Path(path)
    if not runtime_path.exists():
        return None

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if "inference_time_seconds" not in runtime:
        raise ValueError(f"{runtime_path} must contain inference_time_seconds")
    if float(runtime["inference_time_seconds"]) < 0:
        raise ValueError("inference_time_seconds cannot be negative")
    if "training_time_seconds" in runtime and float(runtime["training_time_seconds"]) < 0:
        raise ValueError("training_time_seconds cannot be negative")
    if int(runtime.get("num_test_images", expected_samples)) != expected_samples:
        raise ValueError(
            f"runtime num_test_images={runtime.get('num_test_images')} does not match "
            f"the {expected_samples} evaluated predictions"
        )
    if "hardware" not in runtime or "software" not in runtime:
        raise ValueError(f"{runtime_path} must contain hardware and software objects")
    return runtime


def load_evaluation_data(
    predictions_path: str | Path,
    scores_path: str | Path,
    *,
    expected_num_classes: int | None = 500,
    expected_split: str = "test",
    strict_argmax: bool = True,
) -> EvaluationData:
    predictions_path = Path(predictions_path)
    scores_path = Path(scores_path)

    predictions = pd.read_csv(predictions_path, dtype={"image_id": "string"})
    missing_columns = [column for column in PREDICTION_COLUMNS if column not in predictions]
    if missing_columns:
        raise ValueError(f"{predictions_path} is missing columns: {missing_columns}")
    if predictions.empty:
        raise ValueError(f"{predictions_path} contains no predictions")

    predictions = predictions.loc[:, list(PREDICTION_COLUMNS)].copy()
    if predictions.isna().any().any():
        missing = predictions.columns[predictions.isna().any()].tolist()
        raise ValueError(f"{predictions_path} contains missing values in {missing}")
    predictions["image_id"] = predictions["image_id"].astype(str)
    if predictions["image_id"].duplicated().any():
        duplicates = (
            predictions.loc[predictions["image_id"].duplicated(), "image_id"].head().tolist()
        )
        raise ValueError(f"duplicate image_id values in predictions: {duplicates}")

    for column in ("true_label", "pred_label"):
        numeric = pd.to_numeric(predictions[column], errors="raise")
        if not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError(f"{column} must contain integer class_idx values")
        predictions[column] = numeric.astype(np.int64)
    predictions["top1_score"] = pd.to_numeric(predictions["top1_score"], errors="raise")
    if not np.isfinite(predictions["top1_score"].to_numpy()).all():
        raise ValueError("top1_score contains NaN or infinity")

    methods = predictions["method_name"].astype(str).unique()
    if len(methods) != 1:
        raise ValueError(
            f"one predictions.csv may contain only one method_name; found {methods.tolist()}"
        )
    splits = predictions["split"].astype(str).unique()
    if splits.tolist() != [expected_split]:
        raise ValueError(f"split must be {expected_split!r}; found {splits.tolist()}")

    with np.load(scores_path, allow_pickle=False) as archive:
        missing_keys = [key for key in SCORES_KEYS if key not in archive]
        if missing_keys:
            raise ValueError(f"{scores_path} is missing arrays: {missing_keys}")
        score_image_ids = np.asarray(archive["image_ids"])
        scores = np.asarray(archive["scores"])
        class_indices = np.asarray(archive["class_indices"])

    if scores.ndim != 2:
        raise ValueError(f"scores must be a 2-D [N, C] matrix; got shape {scores.shape}")
    if score_image_ids.ndim != 1 or class_indices.ndim != 1:
        raise ValueError("image_ids and class_indices must both be 1-D arrays")
    if len(score_image_ids) != scores.shape[0]:
        raise ValueError("len(image_ids) must equal scores.shape[0]")
    if len(class_indices) != scores.shape[1]:
        raise ValueError("len(class_indices) must equal scores.shape[1]")
    if not np.issubdtype(scores.dtype, np.number) or not np.isfinite(scores).all():
        raise ValueError("scores must be numeric and contain no NaN or infinity")

    if not np.issubdtype(class_indices.dtype, np.integer):
        if not np.all(np.equal(class_indices, np.floor(class_indices))):
            raise ValueError("class_indices must contain integer values")
    class_indices = class_indices.astype(np.int64)
    if len(np.unique(class_indices)) != len(class_indices):
        raise ValueError("class_indices contains duplicate classes")

    labels = np.sort(class_indices)
    if expected_num_classes is not None:
        expected_labels = np.arange(expected_num_classes, dtype=np.int64)
        if not np.array_equal(labels, expected_labels):
            raise ValueError(
                "class_indices must cover every class_idx from 0 to "
                f"{expected_num_classes - 1}; got {labels.tolist()[:10]}..."
            )

    valid_labels = set(labels.tolist())
    observed_labels = set(predictions["true_label"]) | set(predictions["pred_label"])
    if not observed_labels.issubset(valid_labels):
        raise ValueError(
            f"predictions contain labels absent from class_indices: {sorted(observed_labels - valid_labels)}"
        )

    normalised_score_ids = np.asarray([_normalise_id(item) for item in score_image_ids], dtype=str)
    if len(np.unique(normalised_score_ids)) != len(normalised_score_ids):
        raise ValueError("scores.npz image_ids contains duplicates")
    prediction_ids = predictions["image_id"].to_numpy(dtype=str)
    prediction_set = set(prediction_ids)
    score_set = set(normalised_score_ids)
    if prediction_set != score_set:
        missing_scores = sorted(prediction_set - score_set)[:5]
        extra_scores = sorted(score_set - prediction_set)[:5]
        raise ValueError(
            "predictions.csv and scores.npz image_id sets differ; "
            f"missing scores={missing_scores}, extra scores={extra_scores}"
        )

    score_row_by_id = {image_id: row for row, image_id in enumerate(normalised_score_ids)}
    aligned_rows = np.fromiter(
        (score_row_by_id[image_id] for image_id in prediction_ids), dtype=np.int64
    )
    aligned_scores = scores[aligned_rows]

    score_argmax_labels = class_indices[np.argmax(aligned_scores, axis=1)]
    mismatch_mask = score_argmax_labels != predictions["pred_label"].to_numpy(dtype=np.int64)
    if strict_argmax and mismatch_mask.any():
        examples = predictions.loc[mismatch_mask, ["image_id", "pred_label"]].head().copy()
        examples["scores_argmax_label"] = score_argmax_labels[mismatch_mask][: len(examples)]
        raise ValueError(
            "pred_label does not match the scores argmax after class_indices mapping; examples: "
            f"{examples.to_dict(orient='records')}"
        )

    return EvaluationData(
        predictions=predictions,
        scores=aligned_scores,
        class_indices=class_indices,
        labels=labels,
        method_name=str(methods[0]),
        split=expected_split,
    )
