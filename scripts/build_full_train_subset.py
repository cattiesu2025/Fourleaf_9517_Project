#!/usr/bin/env python3
"""Build metadata for the full iNaturalist 2021 train subset.

The selected 500-class mapping and the existing validation split are preserved.
Images already present in ``train_mini`` are reused; only additional images are
written to the tar extraction path list.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:
    import ijson
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "Missing dependency 'ijson'. Install the project requirements first:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc


CSV_COLUMNS = (
    "image_id",
    "image_path",
    "original_class_id",
    "class_idx",
    "class_name",
    "split",
)


@dataclass(frozen=True)
class SelectedClass:
    original_class_id: int
    class_idx: int
    class_name: str
    category: str


@dataclass(frozen=True)
class FullImage:
    source_image_id: int
    original_class_id: int
    archive_path: str
    canonical_key: str


def read_csv(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(required_columns) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Metadata file is empty: {path}")
    return rows


def canonical_key(image_path: str) -> str:
    """Return ``category-directory/image.jpg`` for an iNaturalist split path."""

    parts = PurePosixPath(image_path.replace("\\", "/")).parts
    marker_positions = [
        parts.index(marker)
        for marker in ("train", "train_mini", "val")
        if marker in parts
    ]
    if not marker_positions:
        raise ValueError(
            f"Expected a train/, train_mini/, or val/ component in path: {image_path}"
        )

    marker_index = min(marker_positions)
    relative_parts = parts[marker_index + 1 :]
    if len(relative_parts) != 2:
        raise ValueError(
            "Expected iNaturalist path category-directory/image.jpg after "
            f"train marker: {image_path}"
        )
    return "/".join(relative_parts)


def safe_archive_path(image_path: str) -> str:
    path = PurePosixPath(image_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path in train.json: {image_path}")
    if len(path.parts) != 3 or path.parts[0] != "train":
        raise ValueError(
            "Expected full-train archive path train/category/image.jpg, found: "
            f"{image_path}"
        )
    return path.as_posix()


def load_selected_classes(path: Path) -> dict[int, SelectedClass]:
    rows = read_csv(
        path,
        ("class_idx", "original_class_id", "class_name", "category"),
    )
    selected: dict[int, SelectedClass] = {}
    class_indices: set[int] = set()

    for row in rows:
        original_id = int(row["original_class_id"])
        class_idx = int(row["class_idx"])
        if original_id in selected:
            raise ValueError(f"Duplicate original_class_id in {path}: {original_id}")
        if class_idx in class_indices:
            raise ValueError(f"Duplicate class_idx in {path}: {class_idx}")
        selected[original_id] = SelectedClass(
            original_class_id=original_id,
            class_idx=class_idx,
            class_name=row["class_name"],
            category=row["category"],
        )
        class_indices.add(class_idx)

    expected_indices = set(range(len(selected)))
    if class_indices != expected_indices:
        raise ValueError(
            f"{path} class_idx values must be consecutive 0..{len(selected) - 1}"
        )
    return selected


def load_split_by_key(
    path: Path,
    selected: dict[int, SelectedClass],
) -> tuple[dict[str, dict[str, str]], set[str]]:
    rows = read_csv(path, CSV_COLUMNS)
    by_key: dict[str, dict[str, str]] = {}
    image_ids: set[str] = set()

    for row in rows:
        original_id = int(row["original_class_id"])
        class_idx = int(row["class_idx"])
        if original_id not in selected:
            raise ValueError(f"{path} contains unselected class {original_id}")
        if selected[original_id].class_idx != class_idx:
            raise ValueError(
                f"{path} class mapping mismatch for original class {original_id}"
            )

        key = canonical_key(row["image_path"])
        if key in by_key:
            raise ValueError(f"{path} contains duplicate image path key: {key}")
        by_key[key] = row

        image_id = str(row["image_id"])
        if image_id in image_ids:
            raise ValueError(f"{path} contains duplicate image_id: {image_id}")
        image_ids.add(image_id)

    return by_key, image_ids


def collect_selected_annotations(
    train_json: Path,
    selected_class_ids: set[int],
) -> dict[int, int]:
    """Map selected full-train image IDs to original category IDs."""

    selected_images: dict[int, int] = {}
    scanned = 0
    with train_json.open("rb") as handle:
        for annotation in ijson.items(handle, "annotations.item"):
            scanned += 1
            category_id = int(annotation["category_id"])
            if category_id not in selected_class_ids:
                continue
            image_id = int(annotation["image_id"])
            previous = selected_images.setdefault(image_id, category_id)
            if previous != category_id:
                raise ValueError(
                    f"Image {image_id} has conflicting selected annotations"
                )

    if not selected_images:
        raise ValueError(
            f"No annotations matched the selected classes in {train_json}"
        )
    print(
        f"Scanned {scanned:,} annotations; "
        f"selected {len(selected_images):,} images."
    )
    return selected_images


def collect_selected_images(
    train_json: Path,
    category_by_image_id: dict[int, int],
) -> list[FullImage]:
    selected_images: list[FullImage] = []
    found_ids: set[int] = set()
    scanned = 0

    with train_json.open("rb") as handle:
        for image in ijson.items(handle, "images.item"):
            scanned += 1
            source_image_id = int(image["id"])
            category_id = category_by_image_id.get(source_image_id)
            if category_id is None:
                continue
            archive_path = safe_archive_path(str(image["file_name"]))
            selected_images.append(
                FullImage(
                    source_image_id=source_image_id,
                    original_class_id=category_id,
                    archive_path=archive_path,
                    canonical_key=canonical_key(archive_path),
                )
            )
            found_ids.add(source_image_id)

    missing_ids = set(category_by_image_id) - found_ids
    if missing_ids:
        examples = sorted(missing_ids)[:5]
        raise ValueError(
            f"{len(missing_ids)} selected annotations have no image entry; "
            f"examples: {examples}"
        )
    print(
        f"Scanned {scanned:,} image records; "
        f"resolved {len(selected_images):,} selected paths."
    )
    return selected_images


def next_numeric_image_id(
    split_id_sets: Iterable[set[str]],
) -> int:
    all_ids: set[str] = set()
    numeric_ids: list[int] = []
    for id_set in split_id_sets:
        overlap = all_ids & id_set
        if overlap:
            raise ValueError(
                "Existing train/val/test image_id values are not globally unique; "
                f"examples: {sorted(overlap)[:5]}"
            )
        all_ids.update(id_set)
        for image_id in id_set:
            try:
                numeric_ids.append(int(image_id))
            except ValueError:
                continue
    return max(numeric_ids, default=-1) + 1


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_full_train_subset(
    *,
    train_json: Path,
    selected_classes_csv: Path,
    current_train_csv: Path,
    validation_csv: Path,
    test_csv: Path,
    output_csv: Path,
    paths_output: Path,
    config_output: Path,
    raw_root: Path,
) -> dict[str, Any]:
    if not train_json.is_file():
        raise FileNotFoundError(
            f"Full train annotation not found: {train_json}\n"
            "Download and extract train.json.tar.gz before running this script."
        )

    selected = load_selected_classes(selected_classes_csv)
    current_train, train_ids = load_split_by_key(current_train_csv, selected)
    validation, validation_ids = load_split_by_key(validation_csv, selected)
    _, test_ids = load_split_by_key(test_csv, selected)

    overlap = set(current_train) & set(validation)
    if overlap:
        raise ValueError(
            "Current train and validation paths overlap; "
            f"examples: {sorted(overlap)[:5]}"
        )

    category_by_image_id = collect_selected_annotations(
        train_json, set(selected)
    )
    full_images = collect_selected_images(train_json, category_by_image_id)

    full_by_key: dict[str, FullImage] = {}
    for image in full_images:
        if image.canonical_key in full_by_key:
            raise ValueError(
                f"Duplicate selected full-train path: {image.canonical_key}"
            )
        full_by_key[image.canonical_key] = image

    missing_current = set(current_train) - set(full_by_key)
    missing_validation = set(validation) - set(full_by_key)
    if missing_current or missing_validation:
        raise ValueError(
            "Existing mini split is not fully represented in full train: "
            f"{len(missing_current)} train and {len(missing_validation)} validation "
            "paths are missing."
        )

    next_image_id = next_numeric_image_id((train_ids, validation_ids, test_ids))
    output_rows: list[dict[str, Any]] = []
    archive_paths: list[str] = []
    reused_count = 0
    excluded_count = 0

    for image in sorted(
        full_images,
        key=lambda item: (
            selected[item.original_class_id].class_idx,
            item.archive_path,
        ),
    ):
        class_info = selected[image.original_class_id]

        validation_row = validation.get(image.canonical_key)
        if validation_row is not None:
            if int(validation_row["original_class_id"]) != image.original_class_id:
                raise ValueError(
                    f"Validation label mismatch for {image.canonical_key}"
                )
            excluded_count += 1
            continue

        current_row = current_train.get(image.canonical_key)
        if current_row is not None:
            if int(current_row["original_class_id"]) != image.original_class_id:
                raise ValueError(
                    f"Current-train label mismatch for {image.canonical_key}"
                )
            image_id: int | str = current_row["image_id"]
            image_path = current_row["image_path"]
            reused_count += 1
        else:
            image_id = next_image_id
            next_image_id += 1
            image_path = (raw_root / image.archive_path).as_posix()
            archive_paths.append(image.archive_path)

        output_rows.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "original_class_id": image.original_class_id,
                "class_idx": class_info.class_idx,
                "class_name": class_info.class_name,
                "split": "train",
            }
        )

    if reused_count != len(current_train):
        raise ValueError(
            f"Expected to reuse {len(current_train):,} current train images, "
            f"reused {reused_count:,}"
        )
    if excluded_count != len(validation):
        raise ValueError(
            f"Expected to exclude {len(validation):,} validation images, "
            f"excluded {excluded_count:,}"
        )

    output_image_ids = [str(row["image_id"]) for row in output_rows]
    if len(output_image_ids) != len(set(output_image_ids)):
        raise ValueError("Generated train_full.csv image_id values are not unique")

    train_counts = Counter(int(row["class_idx"]) for row in output_rows)
    missing_classes = set(range(len(selected))) - set(train_counts)
    if missing_classes:
        raise ValueError(
            f"Generated training split has empty classes: {sorted(missing_classes)}"
        )

    raw_counts = Counter(
        selected[image.original_class_id].class_idx for image in full_images
    )
    per_class_counts = list(train_counts.values())
    config: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "iNaturalist-2021",
        "source_split": "full_train",
        "num_classes": len(selected),
        "selected_classes_csv": selected_classes_csv.as_posix(),
        "current_train_csv": current_train_csv.as_posix(),
        "validation_csv": validation_csv.as_posix(),
        "test_csv": test_csv.as_posix(),
        "full_train_annotation": train_json.as_posix(),
        "class_mapping_preserved": True,
        "validation_excluded_from_training": True,
        "raw_selected_images": len(full_images),
        "excluded_validation_images": excluded_count,
        "training_images": len(output_rows),
        "reused_train_mini_images": reused_count,
        "additional_images_to_extract": len(archive_paths),
        "train_images_per_class": {
            "min": min(per_class_counts),
            "mean": statistics.fmean(per_class_counts),
            "max": max(per_class_counts),
        },
        "raw_images_per_class": {
            "min": min(raw_counts.values()),
            "mean": statistics.fmean(raw_counts.values()),
            "max": max(raw_counts.values()),
        },
    }

    atomic_write_csv(output_csv, output_rows)
    atomic_write_text(
        paths_output,
        "".join(f"{archive_path}\n" for archive_path in archive_paths),
    )
    atomic_write_text(
        config_output,
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train_full.csv for the existing selected classes while "
            "preserving the current validation and test sets."
        )
    )
    parser.add_argument("--train-json", type=Path, default=Path("data/raw/train.json"))
    parser.add_argument(
        "--selected-classes",
        type=Path,
        default=Path("data/metadata/selected_classes.csv"),
    )
    parser.add_argument(
        "--current-train",
        type=Path,
        default=Path("data/metadata/train.csv"),
    )
    parser.add_argument(
        "--validation", type=Path, default=Path("data/metadata/val.csv")
    )
    parser.add_argument("--test", type=Path, default=Path("data/metadata/test.csv"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/metadata/train_full.csv"),
    )
    parser.add_argument(
        "--paths-output",
        type=Path,
        default=Path("data/metadata/full_train_paths.txt"),
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("data/metadata/split_config_full.json"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_full_train_subset(
        train_json=args.train_json,
        selected_classes_csv=args.selected_classes,
        current_train_csv=args.current_train,
        validation_csv=args.validation,
        test_csv=args.test,
        output_csv=args.output_csv,
        paths_output=args.paths_output,
        config_output=args.config_output,
        raw_root=args.raw_root,
    )
    print("\nFull-train subset metadata created:")
    print(f"  training rows:        {config['training_images']:,}")
    print(f"  reused mini images:   {config['reused_train_mini_images']:,}")
    print(f"  validation excluded:  {config['excluded_validation_images']:,}")
    print(f"  images to extract:    {config['additional_images_to_extract']:,}")


if __name__ == "__main__":
    main()
