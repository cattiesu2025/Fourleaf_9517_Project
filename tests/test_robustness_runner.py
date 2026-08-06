from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.evaluation.robustness import (
    ModelPredictor,
    RobustnessSample,
    evaluate_robustness,
    run_clean_inference,
    run_robustness_inference,
)


class DummyColourPredictor(ModelPredictor):
    method_name = "dummy_colour_predictor"
    class_indices = np.asarray([2, 0, 1], dtype=np.int64)

    def predict_scores(self, image: Image.Image) -> np.ndarray:
        red, green, blue = np.asarray(image, dtype=float).mean(axis=(0, 1))
        return np.asarray([blue, red, green])


class DummyBatchColourPredictor(DummyColourPredictor):
    method_name = "dummy_batch_colour_predictor"

    def __init__(self) -> None:
        self.batch_calls = 0
        self.run_starts = 0

    def begin_run(self, degradation_type: str, severity: int) -> None:
        del degradation_type, severity
        self.run_starts += 1

    def predict_scores(self, image: Image.Image) -> np.ndarray:
        raise AssertionError("the batch interface should be preferred")

    def predict_scores_batch(self, images: list[Image.Image], image_ids: list[str]) -> np.ndarray:
        del image_ids
        self.batch_calls += 1
        rows = []
        for image in images:
            red, green, blue = np.asarray(image, dtype=float).mean(axis=(0, 1))
            rows.append([blue, red, green])
        return np.asarray(rows)


def test_dummy_predictor_runs_complete_robustness_matrix(tmp_path: Path) -> None:
    colours = [(240, 10, 10), (10, 240, 10), (10, 10, 240)]
    samples: list[RobustnessSample] = []
    for class_idx, colour in enumerate(colours):
        image_path = tmp_path / f"class_{class_idx}.png"
        Image.new("RGB", (16, 16), colour).save(image_path)
        samples.append(RobustnessSample(f"image-{class_idx}", class_idx, image_path))

    robustness_root = tmp_path / "robustness"
    run_dirs = run_robustness_inference(DummyColourPredictor(), samples, robustness_root)
    assert len(run_dirs) == 20
    for run_dir in run_dirs:
        assert (run_dir / "predictions.csv").is_file()
        assert (run_dir / "scores.npz").is_file()
        assert (run_dir / "runtime.json").is_file()

    summary = evaluate_robustness(
        robustness_root,
        tmp_path / "evaluation",
        methods=[DummyColourPredictor.method_name],
        expected_num_classes=3,
    )
    assert len(summary) == 20
    assert set(summary["degradation_type"]) == {
        "gaussian_noise",
        "blur",
        "brightness",
        "jpeg_compression",
    }


def test_runner_prefers_batch_prediction_and_resets_each_run(tmp_path: Path) -> None:
    colours = [(240, 10, 10), (10, 240, 10), (10, 10, 240)]
    samples = []
    for class_idx, colour in enumerate(colours):
        image_path = tmp_path / f"batch-class-{class_idx}.png"
        Image.new("RGB", (16, 16), colour).save(image_path)
        samples.append(RobustnessSample(str(class_idx), class_idx, image_path))

    predictor = DummyBatchColourPredictor()
    run_robustness_inference(
        predictor,
        samples,
        tmp_path / "batch-robustness",
        batch_size=2,
    )
    assert predictor.run_starts == 20
    assert predictor.batch_calls == 40


def test_evaluator_adds_clean_reference_and_degradation_drops(tmp_path: Path) -> None:
    colours = [(240, 10, 10), (10, 240, 10), (10, 10, 240)]
    samples = []
    for class_idx, colour in enumerate(colours):
        image_path = tmp_path / f"clean-class-{class_idx}.png"
        Image.new("RGB", (16, 16), colour).save(image_path)
        samples.append(RobustnessSample(str(class_idx), class_idx, image_path))

    robustness_root = tmp_path / "clean-reference-robustness"
    run_robustness_inference(DummyColourPredictor(), samples, robustness_root)
    clean_results = tmp_path / "all_results.csv"
    pd.DataFrame(
        [
            {
                "run_id": DummyColourPredictor.method_name,
                "top1_accuracy": 1.0,
                "macro_f1": 1.0,
            }
        ]
    ).to_csv(clean_results, index=False)

    summary = evaluate_robustness(
        robustness_root,
        tmp_path / "clean-reference-evaluation",
        methods=[DummyColourPredictor.method_name],
        expected_num_classes=3,
        clean_results=clean_results,
    )
    assert set(["clean_top1_accuracy", "clean_macro_f1", "top1_drop", "macro_f1_drop"]).issubset(
        summary.columns
    )
    assert (summary["clean_top1_accuracy"] == 1.0).all()
    assert np.allclose(summary["top1_drop"], 1.0 - summary["top1_accuracy"])


def test_evaluator_uses_same_environment_clean_run(tmp_path: Path) -> None:
    colours = [(240, 10, 10), (10, 240, 10), (10, 10, 240)]
    samples = []
    for class_idx, colour in enumerate(colours):
        image_path = tmp_path / f"local-clean-class-{class_idx}.png"
        Image.new("RGB", (16, 16), colour).save(image_path)
        samples.append(RobustnessSample(str(class_idx), class_idx, image_path))

    robustness_root = tmp_path / "local-clean-robustness"
    predictor = DummyColourPredictor()
    clean_dir = run_clean_inference(predictor, samples, robustness_root)
    assert clean_dir == robustness_root / predictor.method_name / "clean" / "severity_0"
    run_robustness_inference(predictor, samples, robustness_root)
    summary = evaluate_robustness(
        robustness_root,
        tmp_path / "local-clean-evaluation",
        methods=[predictor.method_name],
        expected_num_classes=3,
    )
    assert (summary["clean_top1_accuracy"] == 1.0).all()
    assert (summary["clean_macro_f1"] == 1.0).all()
