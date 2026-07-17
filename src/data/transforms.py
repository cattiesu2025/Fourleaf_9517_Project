"""Image transforms shared by scratch and transfer-learning CNN pipelines."""

from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Return deterministic preprocessing for validation and test images."""
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_transform(
    transform_type: str = "none",
    image_size: int = 224,
    is_train: bool = True,
) -> transforms.Compose:
    """Build the transform pipeline requested by the augmentation ablation.

    Parameters
    ----------
    transform_type:
        One of ``none``, ``basic_aug`` or ``strong_aug``.
    image_size:
        Final square CNN input size.
    is_train:
        Random augmentation is only enabled when this is ``True``. Validation
        and test callers should pass ``False`` and will receive deterministic
        preprocessing regardless of ``transform_type``.
    """
    if transform_type not in {"none", "basic_aug", "strong_aug"}:
        raise ValueError(
            "transform_type must be one of: none, basic_aug, strong_aug; "
            f"got {transform_type!r}"
        )

    if not is_train or transform_type == "none":
        return get_eval_transform(image_size=image_size)

    if transform_type == "basic_aug":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.7, 1.0),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    strong_ops = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.6, 1.0),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
    ]
    if hasattr(transforms, "RandAugment"):
        strong_ops.append(transforms.RandAugment(num_ops=2, magnitude=7))
    strong_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )
    return transforms.Compose(strong_ops)
