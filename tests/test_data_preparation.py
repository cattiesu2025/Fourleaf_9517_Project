from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_dataset import CSV_COLUMNS, build_dataset
from scripts.build_longtail import build_longtail_metadata
from scripts.copy_selected_images import copy_selected_images


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def split_row(image_id: int, image_path: str, class_idx: int, split: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "image_path": image_path,
        "original_class_id": 100 + class_idx,
        "class_idx": class_idx,
        "class_name": f"Species {class_idx}",
        "split": split,
    }


def write_annotations(
    path: Path,
    image_categories: list[tuple[int, int]],
    prefix: str,
) -> None:
    categories = sorted({category_id for _, category_id in image_categories})
    payload = {
        "images": [
            {
                "id": image_id,
                "file_name": f"{prefix}/{category_id}/{image_id}.jpg",
            }
            for image_id, category_id in image_categories
        ],
        "annotations": [
            {
                "id": index,
                "image_id": image_id,
                "category_id": category_id,
            }
            for index, (image_id, category_id) in enumerate(image_categories)
        ],
        "categories": [
            {
                "id": category_id,
                "name": f"Species {category_id}",
                "supercategory": "Test",
            }
            for category_id in categories
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_dataset_writes_fixed_split_and_refuses_accidental_overwrite(
    tmp_path: Path,
) -> None:
    train_json = tmp_path / "train.json"
    validation_json = tmp_path / "validation.json"
    output_dir = tmp_path / "metadata"
    write_annotations(
        train_json,
        [(1, 10), (2, 10), (3, 10), (4, 20), (5, 20), (6, 20)],
        "train_mini",
    )
    write_annotations(validation_json, [(7, 10), (8, 20)], "val")

    config = build_dataset(
        train_json=train_json,
        validation_json=validation_json,
        output_dir=output_dir,
        raw_root=Path("data/raw"),
        num_classes=2,
        train_per_class=1,
        validation_per_class=1,
        test_per_class=1,
        seed=500,
    )

    assert config["num_classes"] == 2
    all_rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        with (output_dir / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert {row["split"] for row in rows} == {split}
        assert all(row["image_path"].startswith("data/raw/") for row in rows)
        all_rows.extend(rows)
    assert {int(row["image_id"]) for row in all_rows} == set(range(6))

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        build_dataset(
            train_json=train_json,
            validation_json=validation_json,
            output_dir=output_dir,
            raw_root=Path("data/raw"),
            num_classes=2,
            train_per_class=1,
            validation_per_class=1,
            test_per_class=1,
            seed=500,
        )


def test_build_longtail_metadata_is_deterministic_and_can_balance_statically(
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    rows = [
        split_row(class_idx * 10 + index, f"data/raw/{class_idx}/{index}.jpg", class_idx, "train")
        for class_idx in range(3)
        for index in range(4)
    ]
    write_csv(train_csv, rows)
    output_dir = tmp_path / "metadata"

    config = build_longtail_metadata(
        train_csv=train_csv,
        output_dir=output_dir,
        seed=500,
        minimum_per_class=1,
        maximum_per_class=4,
        static_oversample=True,
    )
    original = (output_dir / "longtail_train.csv").read_text(encoding="utf-8")
    with (output_dir / "longtail_resampled_train.csv").open(newline="", encoding="utf-8") as handle:
        resampled = list(csv.DictReader(handle))

    assert config["num_classes"] == 3
    assert config["resampling_strategy"] == "static_oversampling"
    assert len(resampled) == 12
    assert {
        sum(int(row["class_idx"]) == class_idx for row in resampled) for class_idx in range(3)
    } == {4}

    build_longtail_metadata(
        train_csv=train_csv,
        output_dir=output_dir,
        seed=500,
        minimum_per_class=1,
        maximum_per_class=4,
        static_oversample=True,
        force=True,
    )
    assert (output_dir / "longtail_train.csv").read_text(encoding="utf-8") == original


def test_copy_selected_images_supports_dry_run_and_preserves_relative_paths(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    metadata_dir = tmp_path / "metadata"
    output_dir = tmp_path / "subset"
    image_paths = {
        "train": "data/raw/train/class-a/one.jpg",
        "val": "data/raw/train/class-a/two.jpg",
        "test": "data/raw/val/class-a/three.jpg",
    }
    for index, (split, image_path) in enumerate(image_paths.items()):
        source = source_root / image_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"image-{index}".encode())
        write_csv(metadata_dir / f"{split}.csv", [split_row(index, image_path, 0, split)])

    preview = copy_selected_images(
        metadata_dir=metadata_dir,
        source_root=source_root,
        output_dir=output_dir,
        dry_run=True,
    )
    assert preview.available_images == 3
    assert preview.copied_images == 0
    assert not output_dir.exists()

    result = copy_selected_images(
        metadata_dir=metadata_dir,
        source_root=source_root,
        output_dir=output_dir,
    )
    assert result.copied_images == 3
    assert (output_dir / "train/class-a/one.jpg").read_bytes() == b"image-0"
    assert (output_dir / "val/class-a/three.jpg").read_bytes() == b"image-2"
