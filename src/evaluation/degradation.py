from __future__ import annotations

import hashlib
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "robustness.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _parameter_value(degradation_type: str, severity: int) -> float | int:
    config = _load_config()
    levels = config["severity_levels"]
    if severity not in levels:
        raise ValueError(f"severity must be one of {levels}; got {severity}")
    try:
        values = config["degradations"][degradation_type]["values"]
    except KeyError as exc:
        choices = ", ".join(config["degradations"])
        raise ValueError(
            f"unknown degradation_type {degradation_type!r}; choose from {choices}"
        ) from exc
    return values[levels.index(severity)]


def degradation_seed(image_id: str | int, base_seed: int = 9517) -> int:
    """Return a stable per-image seed independent of Python's hash randomisation."""

    digest = hashlib.blake2b(
        str(image_id).encode("utf-8"),
        digest_size=8,
        person=b"COMP9517",
    ).digest()
    return (base_seed + int.from_bytes(digest, byteorder="little")) % (2**32)


def apply_degradation(
    image: Image.Image,
    degradation_type: str,
    severity: int,
    seed: int | None = None,
) -> Image.Image:
    """Apply one configured degradation to an unprocessed RGB image.

    This function must run immediately after image loading and before any
    model-specific resize, greyscale conversion, feature extraction, or
    normalisation. The returned image is always an RGB PIL image.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    rgb = image.convert("RGB")
    value = _parameter_value(degradation_type, severity)

    if degradation_type == "gaussian_noise":
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        rng = np.random.default_rng(seed)
        noisy = np.clip(array + rng.normal(0.0, float(value), array.shape), 0.0, 1.0)
        return Image.fromarray(np.rint(noisy * 255.0).astype(np.uint8), mode="RGB")

    if degradation_type == "blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=float(value)))

    if degradation_type == "brightness":
        return ImageEnhance.Brightness(rgb).enhance(float(value)).convert("RGB")

    if degradation_type == "jpeg_compression":
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=int(value))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()

    raise AssertionError("configuration validation should reject unknown degradations")
