from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.contracts import load_evaluation_data
from src.evaluation.evaluate import evaluate_method


def _write_fixture(root: Path, *, mismatch_argmax: bool = False) -> tuple[Path, Path, Path, Path]:
    image_ids = np.asarray(["img-0", "img-1", "img-2", "img-3", "img-4", "img-5"])
    true_labels = np.asarray([0, 1, 2, 0, 1, 2])
    class_indices = np.asarray([2, 0, 1])
    scores = np.asarray(
        [
            [0.1, 2.0, 0.2],
            [0.2, 0.1, 2.1],
            [2.2, 0.2, 0.1],
            [0.3, 1.8, 0.5],
            [0.2, 1.4, 1.2],
            [1.5, 0.2, 0.4],
        ]
    )
    pred_labels = class_indices[np.argmax(scores, axis=1)]
    if mismatch_argmax:
        pred_labels[0] = 1
    predictions = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_label": true_labels,
            "pred_label": pred_labels,
            "top1_score": scores.max(axis=1),
            "method_name": "fixture_method",
            "split": "test",
        }
    )
    predictions_path = root / "predictions.csv"
    scores_path = root / "scores.npz"
    classes_path = root / "idx_to_class.json"
    runtime_path = root / "runtime.json"
    root.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    order = np.asarray([4, 2, 0, 5, 1, 3])
    np.savez(
        scores_path,
        image_ids=image_ids[order],
        scores=scores[order],
        class_indices=class_indices,
    )
    classes_path.write_text(
        json.dumps({str(i): {"class_name": f"Class {i}"} for i in range(3)}),
        encoding="utf-8",
    )
    runtime_path.write_text(
        json.dumps(
            {
                "training_time_seconds": 10,
                "inference_time_seconds": 1,
                "num_test_images": 6,
                "hardware": {"platform": "test", "cpu": "test", "gpu": None},
                "software": {"python": "3.11"},
            }
        ),
        encoding="utf-8",
    )
    return predictions_path, scores_path, classes_path, runtime_path


def test_loader_aligns_image_ids_and_maps_permuted_class_columns(
    tmp_path: Path,
) -> None:
    predictions_path, scores_path, _, _ = _write_fixture(tmp_path)
    data = load_evaluation_data(predictions_path, scores_path, expected_num_classes=3)
    np.testing.assert_array_equal(data.class_indices, [2, 0, 1])
    np.testing.assert_array_equal(data.y_pred, data.class_indices[np.argmax(data.scores, axis=1)])


def test_argmax_mismatch_is_rejected(tmp_path: Path) -> None:
    predictions_path, scores_path, _, _ = _write_fixture(tmp_path, mismatch_argmax=True)
    with pytest.raises(ValueError, match="argmax"):
        load_evaluation_data(predictions_path, scores_path, expected_num_classes=3)


def test_evaluate_method_writes_complete_analysis(tmp_path: Path) -> None:
    predictions_path, scores_path, classes_path, runtime_path = _write_fixture(tmp_path / "input")
    output_dir = tmp_path / "evaluation"
    payload = evaluate_method(
        predictions_path,
        scores_path,
        output_dir,
        classes_path=classes_path,
        runtime_path=runtime_path,
        expected_num_classes=3,
    )
    assert payload["metrics"]["top1_accuracy"] == pytest.approx(5 / 6)
    assert payload["metrics"]["top5_accuracy"] == 1.0
    expected_files = {
        "metrics.json",
        "confusion_matrix.npy",
        "confusion_matrix.png",
        "confusion_matrix_top_classes.png",
        "per_class_metrics.csv",
        "top_confused_pairs.csv",
        "failure_cases.csv",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})
    failures = pd.read_csv(output_dir / "failure_cases.csv")
    assert len(failures) == 1
