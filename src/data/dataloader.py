"""DataLoader factory aligned with the project metadata contract."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import INatDataset
from src.data.sampling import make_weighted_sampler
from src.data.transforms import get_transform


def resolve_csv_path(
    split: str,
    csv_file: str | Path | None = None,
    metadata_dir: str | Path = "data/metadata",
) -> Path:
    """Resolve a split name to a metadata CSV path."""
    if csv_file is not None:
        return Path(csv_file)

    split_to_file = {
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv",
        "longtail_train": "longtail_train.csv",
    }
    if split not in split_to_file:
        raise ValueError(
            f"Unknown split {split!r}. Pass csv_file explicitly or use one of "
            f"{sorted(split_to_file)}."
        )
    return Path(metadata_dir) / split_to_file[split]


def get_dataloader(
    split: str,
    transform_type: str = "none",
    csv_file: str | Path | None = None,
    metadata_dir: str | Path = "data/metadata",
    batch_size: int = 64,
    shuffle: bool | None = None,
    sampler: str = "none",
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool | None = None,
    seed: int | None = None,
    return_dict: bool = False,
    degradation_type: str | None = None,
    severity: int | None = None,
) -> DataLoader:
    """Build a PyTorch DataLoader for train/val/test metadata CSV files."""
    csv_path = resolve_csv_path(split=split, csv_file=csv_file, metadata_dir=metadata_dir)
    is_train = split in {"train", "longtail_train"}
    effective_transform = transform_type if is_train else "none"
    transform = get_transform(
        transform_type=effective_transform,
        image_size=image_size,
        is_train=is_train,
    )

    dataset = INatDataset(
        csv_file=csv_path,
        transform=transform,
        return_dict=return_dict,
        degradation_type=degradation_type,
        severity=severity,
        degradation_seed=seed,
    )

    weighted_sampler = None
    if sampler not in {"none", "weighted_random"}:
        raise ValueError("sampler must be one of: none, weighted_random")
    if sampler == "weighted_random":
        weighted_sampler = make_weighted_sampler(dataset.targets)

    if shuffle is None:
        shuffle = is_train and weighted_sampler is None
    if weighted_sampler is not None:
        shuffle = False

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=weighted_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
