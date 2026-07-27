from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.compare import build_report_artifacts


def _write_run(root: Path, method_name: str, *, wrong_label: bool = False) -> None:
    root.mkdir(parents=True)
    image_ids = np.asarray(["a", "b", "c", "d", "e", "f"])
    true_labels = np.asarray([0, 1, 2, 0, 1, 2])
    if wrong_label:
        true_labels[0] = 1
    class_indices = np.asarray([0, 1, 2])
    scores = np.asarray(
        [
            [3.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 3.0],
            [3.0, 0.0, 1.0],
            [0.0, 3.0, 1.0],
            [1.0, 0.0, 3.0],
        ]
    )
    predictions = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_label": true_labels,
            "pred_label": class_indices[np.argmax(scores, axis=1)],
            "top1_score": scores.max(axis=1),
            "method_name": method_name,
            "split": "test",
        }
    )
    predictions.to_csv(root / "predictions.csv", index=False)
    np.savez(
        root / "scores.npz",
        image_ids=image_ids,
        scores=scores,
        class_indices=class_indices,
    )
    (root / "runtime.json").write_text(
        json.dumps(
            {
                "training_time_seconds": 2.0,
                "inference_time_seconds": 1.0,
                "num_test_images": 6,
                "hardware": {"platform": "test", "cpu": "test", "gpu": None},
                "software": {"python": "3.11"},
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(path: Path, second_run: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {
                        "id": "one",
                        "display_name": "One",
                        "dataset_group": "limited",
                        "tables": ["main"],
                        "predictions": "one/predictions.csv",
                        "scores": "one/scores.npz",
                        "runtime": "one/runtime.json",
                    },
                    {
                        "id": "two",
                        "display_name": "Two",
                        "dataset_group": "limited",
                        "tables": ["main"],
                        "predictions": str(second_run / "predictions.csv"),
                        "scores": str(second_run / "scores.npz"),
                        "runtime": str(second_run / "runtime.json"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_compare_builds_tables_figures_and_shared_test_evidence(tmp_path: Path) -> None:
    _write_run(tmp_path / "one", "method_one")
    _write_run(tmp_path / "two", "method_two")
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, tmp_path / "two")

    output = tmp_path / "report"
    summary = build_report_artifacts(
        manifest, output, artifact_root=tmp_path, expected_num_classes=3
    )
    assert len(summary) == 2
    assert summary["top1_accuracy"].tolist() == [1.0, 1.0]
    assert summary.loc[0, "predictions"] == "one/predictions.csv"
    assert str(tmp_path) not in (output / "all_results.csv").read_text(encoding="utf-8")
    assert {
        "all_results.csv",
        "main.csv",
        "main.tex",
        "main_performance_vs_time.png",
        "main_performance_vs_time.pdf",
        "evaluation_evidence.json",
    }.issubset({path.name for path in output.iterdir()})


def test_compare_rejects_disagreement_in_true_test_labels(tmp_path: Path) -> None:
    _write_run(tmp_path / "one", "method_one")
    _write_run(tmp_path / "two", "method_two", wrong_label=True)
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, tmp_path / "two")

    with pytest.raises(ValueError, match="true_label"):
        build_report_artifacts(
            manifest,
            tmp_path / "report",
            artifact_root=tmp_path,
            expected_num_classes=3,
        )
