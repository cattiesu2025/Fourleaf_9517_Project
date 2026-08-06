"""Dataset implementation backed by the project metadata CSV contract."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "original_class_id",
    "class_idx",
    "class_name",
    "split",
}


class INatDataset(Dataset):
    """iNaturalist subset dataset driven only by metadata CSV rows.

    Each item returns ``(image, label, image_id)`` by default. ``image`` is
    transformed when a transform is provided.
    """

    def __init__(
        self,
        csv_file: str | Path,
        transform: Callable[[Image.Image], Any] | None = None,
        return_dict: bool = False,
        degradation_type: str | None = None,
        severity: int | None = None,
        degradation_seed: int | None = None,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.return_dict = return_dict
        self.degradation_type = degradation_type
        self.severity = severity
        self.degradation_seed = degradation_seed
        self.records = self._read_records(self.csv_file)
        self.targets = [int(row["class_idx"]) for row in self.records]
        self.image_ids = [self._parse_image_id(row["image_id"]) for row in self.records]

        if self.degradation_type is not None and self.severity is None:
            raise ValueError("severity is required when degradation_type is provided")

    @staticmethod
    def _read_records(csv_file: Path) -> list[dict[str, str]]:
        if not csv_file.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_file}")

        with csv_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - fieldnames)
            if missing:
                raise ValueError(f"{csv_file} is missing required columns: {missing}")
            rows = list(reader)

        if not rows:
            raise ValueError(f"Metadata CSV is empty: {csv_file}")
        return rows

    @staticmethod
    def _parse_image_id(value: str) -> int | str:
        try:
            return int(value)
        except ValueError:
            return value

    @staticmethod
    def _load_degradation_fn() -> Callable[..., Image.Image]:
        try:
            from src.evaluation.degradation import apply_degradation
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "degradation_type was requested, but src/evaluation/degradation.py "
                "is not available. Restore that module or run without --degradation."
            ) from exc
        return apply_degradation

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int, int | str] | dict[str, Any]:
        row = self.records[index]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image referenced by {self.csv_file} does not exist: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.degradation_type is not None:
                apply_degradation = self._load_degradation_fn()
                seed = None
                if self.degradation_seed is not None:
                    seed = self.degradation_seed + index
                image = apply_degradation(
                    image,
                    degradation_type=self.degradation_type,
                    severity=int(self.severity),
                    seed=seed,
                )
            if self.transform is not None:
                image = self.transform(image)

        label = int(row["class_idx"])
        image_id = self._parse_image_id(row["image_id"])

        if self.return_dict:
            return {
                "image": image,
                "class_idx": label,
                "image_id": image_id,
                "image_path": row["image_path"],
            }
        return image, label, image_id
