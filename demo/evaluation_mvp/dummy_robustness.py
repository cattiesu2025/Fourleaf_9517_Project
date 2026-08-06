from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.robustness import (
    ModelPredictor,
    RobustnessSample,
    evaluate_robustness,
    run_robustness_inference,
)


class DummyColourPredictor(ModelPredictor):
    """Tiny predictor used only to exercise the shared interface."""

    method_name = "dummy_colour_predictor"
    class_indices = np.asarray([2, 0, 1], dtype=np.int64)

    def predict_scores(self, image: Image.Image) -> np.ndarray:
        red, green, blue = np.asarray(image, dtype=float).mean(axis=(0, 1))
        return np.asarray([blue, red, green])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 4x5 robustness MVP with a dummy predictor."
    )
    parser.add_argument("--output", type=Path, default=Path("tmp/mvp-dummy-robustness"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.force:
            raise SystemExit(f"{args.output} exists; pass --force to replace it")
        shutil.rmtree(args.output)

    image_dir = args.output / "images"
    image_dir.mkdir(parents=True)
    colours = [(240, 10, 10), (10, 240, 10), (10, 10, 240)]
    samples: list[RobustnessSample] = []
    for repetition in range(2):
        for class_idx, colour in enumerate(colours):
            image_id = f"demo-{repetition}-{class_idx}"
            image_path = image_dir / f"{image_id}.png"
            Image.new("RGB", (24, 24), colour).save(image_path)
            samples.append(RobustnessSample(image_id, class_idx, image_path))

    robustness_root = args.output / "outputs" / "robustness"
    run_dirs = run_robustness_inference(DummyColourPredictor(), samples, robustness_root)
    summary = evaluate_robustness(
        robustness_root,
        args.output / "outputs" / "evaluation",
        methods=[DummyColourPredictor.method_name],
        expected_num_classes=3,
    )
    print(
        f"Created {len(run_dirs)} inference runs and {len(summary)} summary rows in {args.output}"
    )


if __name__ == "__main__":
    main()
