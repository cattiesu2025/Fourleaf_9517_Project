import numpy as np
import pytest
from PIL import Image

from src.evaluation.degradation import apply_degradation, degradation_seed


@pytest.fixture
def sample_image() -> Image.Image:
    values = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    return Image.fromarray(values, mode="RGB")


@pytest.mark.parametrize(
    "degradation_type",
    ["gaussian_noise", "blur", "brightness", "jpeg_compression"],
)
@pytest.mark.parametrize("severity", [1, 5])
def test_degradation_preserves_image_contract(
    sample_image: Image.Image,
    degradation_type: str,
    severity: int,
) -> None:
    result = apply_degradation(sample_image, degradation_type, severity, seed=9517)
    assert result.mode == "RGB"
    assert result.size == sample_image.size


def test_noise_is_reproducible(sample_image: Image.Image) -> None:
    first = np.asarray(apply_degradation(sample_image, "gaussian_noise", 3, seed=9517))
    second = np.asarray(apply_degradation(sample_image, "gaussian_noise", 3, seed=9517))
    np.testing.assert_array_equal(first, second)


def test_invalid_severity_is_rejected(sample_image: Image.Image) -> None:
    with pytest.raises(ValueError, match="severity"):
        apply_degradation(sample_image, "blur", 0)


def test_per_image_seed_is_stable_and_distinct() -> None:
    assert degradation_seed("image-1") == degradation_seed("image-1")
    assert degradation_seed("image-1") != degradation_seed("image-2")
