from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate import evaluate_method


def build_demo() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    demo_root = Path("demo/evaluation_mvp")
    input_dir = demo_root / "input"
    expected_dir = demo_root / "expected_output"
    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(expected_dir, ignore_errors=True)
    input_dir.mkdir(parents=True)

    image_ids = np.asarray([f"sample-{index:02d}" for index in range(12)])
    true_labels = np.arange(12, dtype=np.int64) % 3
    class_indices = np.asarray([2, 0, 1], dtype=np.int64)
    rng = np.random.default_rng(9517)
    scores = rng.normal(0, 0.25, (len(image_ids), len(class_indices)))
    column_by_class = {int(label): column for column, label in enumerate(class_indices)}
    for row, label in enumerate(true_labels):
        scores[row, column_by_class[int(label)]] += 1.5
    scores[4, column_by_class[2]] += 2.0
    scores[9, column_by_class[1]] += 2.0
    pred_labels = class_indices[np.argmax(scores, axis=1)]

    pd.DataFrame(
        {
            "image_id": image_ids,
            "true_label": true_labels,
            "pred_label": pred_labels,
            "top1_score": scores.max(axis=1),
            "method_name": "dummy_classifier",
            "split": "test",
        }
    ).to_csv(input_dir / "predictions.csv", index=False)

    row_order = np.asarray([7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4, 6])
    np.savez(
        input_dir / "scores.npz",
        image_ids=image_ids[row_order],
        scores=scores[row_order],
        class_indices=class_indices,
    )
    (input_dir / "idx_to_class.json").write_text(
        json.dumps(
            {str(index): {"class_name": f"Demo class {index}"} for index in range(3)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (input_dir / "runtime.json").write_text(
        json.dumps(
            {
                "training_time_seconds": 1.2,
                "inference_time_seconds": 0.08,
                "num_test_images": len(image_ids),
                "hardware": {"platform": "demo", "cpu": "dummy", "gpu": None},
                "software": {"python": "3.11", "model": "dummy"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    evaluate_method(
        input_dir / "predictions.csv",
        input_dir / "scores.npz",
        expected_dir,
        classes_path=input_dir / "idx_to_class.json",
        runtime_path=input_dir / "runtime.json",
        expected_num_classes=3,
    )
    print(f"MVP demo inputs and expected outputs written to {demo_root}")


if __name__ == "__main__":
    build_demo()
