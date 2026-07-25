from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.scan_dataset_quality import (
    HammingIndex,
    HashItem,
    Thresholds,
    scan_dataset_quality,
)


SOURCE_COLUMNS = (
    "image_id",
    "image_path",
    "original_class_id",
    "class_idx",
    "class_name",
    "split",
)


def write_metadata(
    path: Path,
    rows: list[dict[str, str | int]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def row(
    image_id: str,
    image_path: Path,
    class_idx: int,
    split: str,
) -> dict[str, str | int]:
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "original_class_id": 100 + class_idx,
        "class_idx": class_idx,
        "class_name": f"Species {class_idx}",
        "split": split,
    }


def save_pattern(path: Path, quality: int = 90) -> None:
    y, x = np.mgrid[:64, :64]
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[..., 0] = (x * 4) % 256
    array[..., 1] = (y * 4) % 256
    array[..., 2] = ((x + y) * 2) % 256
    image = Image.fromarray(array, mode="RGB")
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=quality)
    else:
        image.save(path)


def test_quality_scan_excludes_only_safe_automatic_cases(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    validation_exact = image_dir / "validation_exact.jpg"
    save_pattern(validation_exact, quality=88)
    train_holdout_exact = image_dir / "train_holdout_exact.jpg"
    shutil.copyfile(validation_exact, train_holdout_exact)

    train_keep = image_dir / "train_keep.jpg"
    Image.new("RGB", (64, 64), (20, 80, 160)).save(train_keep)
    train_duplicate = image_dir / "train_duplicate.jpg"
    shutil.copyfile(train_keep, train_duplicate)

    conflict_a = image_dir / "conflict_a.jpg"
    Image.new("RGB", (64, 64), (200, 80, 20)).save(conflict_a)
    conflict_b = image_dir / "conflict_b.jpg"
    shutil.copyfile(conflict_a, conflict_b)

    corrupt = image_dir / "corrupt.jpg"
    corrupt.write_bytes(b"not a jpeg")

    dark = image_dir / "dark.jpg"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(dark)
    overexposed = image_dir / "overexposed.jpg"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(overexposed)

    test_near = image_dir / "test_near.png"
    train_near = image_dir / "train_near.bmp"
    save_pattern(test_near)
    save_pattern(train_near)
    assert test_near.read_bytes() != train_near.read_bytes()

    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    write_metadata(
        train_csv,
        [
            row("t0", train_holdout_exact, 0, "train"),
            row("t1", train_keep, 0, "train"),
            row("t2", train_duplicate, 0, "train"),
            row("t3", conflict_a, 0, "train"),
            row("t4", conflict_b, 1, "train"),
            row("t5", corrupt, 0, "train"),
            row("t6", dark, 0, "train"),
            row("t7", overexposed, 1, "train"),
            row("t8", train_near, 1, "train"),
        ],
    )
    write_metadata(
        validation_csv,
        [row("v0", validation_exact, 0, "val")],
    )
    write_metadata(
        test_csv,
        [row("x0", test_near, 1, "test")],
    )

    clean_csv = tmp_path / "clean.csv"
    report_csv = tmp_path / "report.csv"
    summary_json = tmp_path / "summary.json"
    summary = scan_dataset_quality(
        train_csv=train_csv,
        validation_csv=validation_csv,
        test_csv=test_csv,
        clean_csv=clean_csv,
        report_csv=report_csv,
        summary_json=summary_json,
        thresholds=Thresholds(),
        workers=1,
    )

    with clean_csv.open(newline="", encoding="utf-8") as handle:
        clean_rows = list(csv.DictReader(handle))
    assert {item["image_id"] for item in clean_rows} == {
        "t1",
        "t6",
        "t7",
        "t8",
    }

    with report_csv.open(newline="", encoding="utf-8") as handle:
        report = {
            item["image_id"]: item for item in csv.DictReader(handle)
        }
    assert report["t0"]["exclusion_reason"] == "exact_duplicate_holdout"
    assert report["t2"]["exclusion_reason"] == "exact_duplicate_train"
    assert report["t3"]["exclusion_reason"] == (
        "exact_duplicate_label_conflict"
    )
    assert report["t4"]["exclusion_reason"] == (
        "exact_duplicate_label_conflict"
    )
    assert report["t5"]["exclusion_reason"] == "corrupt"
    assert report["t6"]["is_too_dark"] == "True"
    assert report["t6"]["exclude_from_clean"] == "False"
    assert report["t7"]["is_overexposed"] == "True"
    assert report["t7"]["exclude_from_clean"] == "False"
    assert report["t8"]["near_duplicate_scope"] == "test"
    assert report["t8"]["exclude_from_clean"] == "False"

    assert summary["train_input_images"] == 9
    assert summary["train_clean_images"] == 4
    assert summary["train_excluded_images"] == 5


def test_quality_scan_fails_before_writing_when_images_are_missing(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.jpg"
    Image.new("RGB", (32, 32), (100, 100, 100)).save(existing)
    missing = tmp_path / "missing.jpg"

    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    write_metadata(train_csv, [row("t0", missing, 0, "train")])
    write_metadata(validation_csv, [row("v0", existing, 0, "val")])
    write_metadata(test_csv, [row("x0", existing, 0, "test")])

    clean_csv = tmp_path / "clean.csv"
    with pytest.raises(FileNotFoundError, match="Complete extraction"):
        scan_dataset_quality(
            train_csv=train_csv,
            validation_csv=validation_csv,
            test_csv=test_csv,
            clean_csv=clean_csv,
            report_csv=tmp_path / "report.csv",
            summary_json=tmp_path / "summary.json",
            thresholds=Thresholds(),
            workers=1,
        )
    assert not clean_csv.exists()


def test_hamming_index_finds_all_hashes_within_radius() -> None:
    index = HammingIndex(radius=6)
    source = HashItem(
        key="source",
        image_id="1",
        role="validation",
        class_idx=0,
        sha256="a",
        phash=0,
    )
    index.add(source)

    six_bits = sum(1 << bit for bit in (0, 10, 19, 28, 37, 46))
    query = HashItem(
        key="query",
        image_id="2",
        role="train",
        class_idx=0,
        sha256="b",
        phash=six_bits,
    )
    assert index.nearest(query) == (source, 6)

    seven_bits = six_bits | (1 << 55)
    outside = HashItem(
        key="outside",
        image_id="3",
        role="train",
        class_idx=0,
        sha256="c",
        phash=seven_bits,
    )
    assert index.nearest(outside) is None
