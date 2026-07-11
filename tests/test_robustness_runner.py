from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.robustness import (
    ModelPredictor,
    RobustnessSample,
    evaluate_robustness,
    run_robustness_inference,
)


class DummyColourPredictor(ModelPredictor):
    method_name = "dummy_colour_predictor"
    class_indices = np.asarray([2, 0, 1], dtype=np.int64)

    def predict_scores(self, image: Image.Image) -> np.ndarray:
        red, green, blue = np.asarray(image, dtype=float).mean(axis=(0, 1))
        return np.asarray([blue, red, green])


def test_dummy_predictor_runs_complete_robustness_matrix(tmp_path: Path) -> None:
    colours = [(240, 10, 10), (10, 240, 10), (10, 10, 240)]
    samples: list[RobustnessSample] = []
    for class_idx, colour in enumerate(colours):
        image_path = tmp_path / f"class_{class_idx}.png"
        Image.new("RGB", (16, 16), colour).save(image_path)
        samples.append(RobustnessSample(f"image-{class_idx}", class_idx, image_path))

    robustness_root = tmp_path / "robustness"
    run_dirs = run_robustness_inference(
        DummyColourPredictor(), samples, robustness_root
    )
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
