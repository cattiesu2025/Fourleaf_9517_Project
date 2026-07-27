from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path

import yaml


EXPECTED = {
    "torch": "2.3.1",
    "torchvision": "0.18.1",
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "scikit-learn": "1.5.1",
    "opencv-python": "4.10.0.84",
    "Pillow": "10.4.0",
    "matplotlib": "3.9.2",
    "PyYAML": "6.0.2",
    "tqdm": "4.66.5",
    "scipy": "1.14.0",
    "scikit-image": "0.24.0",
    "ijson": "3.3.0",
}

IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "scikit-image": "skimage",
}


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(f"Expected Python 3.11, found {platform.python_version()}")

    mismatches: list[str] = []
    for distribution, expected in EXPECTED.items():
        actual = importlib.metadata.version(distribution)
        importlib.import_module(IMPORT_NAMES.get(distribution, distribution))
        if actual != expected:
            mismatches.append(f"{distribution}: expected {expected}, found {actual}")

    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "robustness.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["severity_levels"] != [1, 2, 3, 4, 5]:
        mismatches.append("robustness severity_levels must be exactly 1..5")
    if set(config["degradations"]) != {
        "gaussian_noise",
        "blur",
        "brightness",
        "jpeg_compression",
    }:
        mismatches.append("robustness config must contain the four agreed degradations")

    if mismatches:
        raise SystemExit("Environment check failed:\n- " + "\n- ".join(mismatches))

    import torch

    accelerator = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Python {platform.python_version()} ({platform.machine()})")
    print(f"PyTorch {torch.__version__}; available accelerator: {accelerator}")
    print("All pinned imports and robustness configuration checks passed.")


if __name__ == "__main__":
    main()
