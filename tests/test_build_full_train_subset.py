from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_full_train_subset import build_full_train_subset

CSV_COLUMNS = (
    "image_id",
    "image_path",
    "original_class_id",
    "class_idx",
    "class_name",
    "split",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def metadata_row(
    image_id: int,
    image_path: str,
    original_class_id: int,
    class_idx: int,
    class_name: str,
    split: str,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "image_path": image_path,
        "original_class_id": original_class_id,
        "class_idx": class_idx,
        "class_name": class_name,
        "split": split,
    }


def prepare_fixture(tmp_path: Path) -> dict[str, Path]:
    selected_path = tmp_path / "selected_classes.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "class_idx",
                "original_class_id",
                "class_name",
                "category",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "class_idx": 0,
                    "original_class_id": 10,
                    "class_name": "Species A",
                    "category": "Plants",
                },
                {
                    "class_idx": 1,
                    "original_class_id": 20,
                    "class_name": "Species B",
                    "category": "Birds",
                },
            ]
        )

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"
    write_csv(
        train_path,
        [
            metadata_row(
                0,
                "data/raw/train_mini/00010_Species_A/a.jpg",
                10,
                0,
                "Species A",
                "train",
            ),
            metadata_row(
                1,
                "data/raw/train_mini/00020_Species_B/e.jpg",
                20,
                1,
                "Species B",
                "train",
            ),
        ],
    )
    write_csv(
        val_path,
        [
            metadata_row(
                2,
                "data/raw/train_mini/00010_Species_A/b.jpg",
                10,
                0,
                "Species A",
                "val",
            ),
            metadata_row(
                3,
                "data/raw/train_mini/00020_Species_B/f.jpg",
                20,
                1,
                "Species B",
                "val",
            ),
        ],
    )
    write_csv(
        test_path,
        [
            metadata_row(
                4,
                "data/raw/val/00010_Species_A/t1.jpg",
                10,
                0,
                "Species A",
                "test",
            ),
            metadata_row(
                5,
                "data/raw/val/00020_Species_B/t2.jpg",
                20,
                1,
                "Species B",
                "test",
            ),
        ],
    )

    images = [
        {"id": 100, "file_name": "train/00010_Species_A/a.jpg"},
        {"id": 101, "file_name": "train/00010_Species_A/b.jpg"},
        {"id": 102, "file_name": "train/00010_Species_A/c.jpg"},
        {"id": 103, "file_name": "train/00010_Species_A/d.jpg"},
        {"id": 104, "file_name": "train/00020_Species_B/e.jpg"},
        {"id": 105, "file_name": "train/00020_Species_B/f.jpg"},
        {"id": 106, "file_name": "train/00020_Species_B/g.jpg"},
        {"id": 107, "file_name": "train/99999_Other/z.jpg"},
    ]
    annotations = [
        {
            "id": index,
            "image_id": image["id"],
            "category_id": 10 if image["id"] < 104 else 20 if image["id"] < 107 else 99,
        }
        for index, image in enumerate(images)
    ]
    train_json_path = tmp_path / "train.json"
    train_json_path.write_text(
        json.dumps({"images": images, "annotations": annotations}),
        encoding="utf-8",
    )
    return {
        "selected": selected_path,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "json": train_json_path,
    }


def run_build(tmp_path: Path, fixture: dict[str, Path]) -> dict[str, object]:
    return build_full_train_subset(
        train_json=fixture["json"],
        selected_classes_csv=fixture["selected"],
        current_train_csv=fixture["train"],
        validation_csv=fixture["val"],
        test_csv=fixture["test"],
        output_csv=tmp_path / "train_full.csv",
        paths_output=tmp_path / "full_train_paths.txt",
        config_output=tmp_path / "split_config_full.json",
        raw_root=Path("data/raw"),
    )


def test_build_reuses_mini_excludes_validation_and_lists_only_new_images(
    tmp_path: Path,
) -> None:
    fixture = prepare_fixture(tmp_path)
    config = run_build(tmp_path, fixture)

    with (tmp_path / "train_full.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 5
    assert {row["image_id"] for row in rows} == {"0", "1", "6", "7", "8"}
    assert not any(row["image_path"].endswith(("/b.jpg", "/f.jpg")) for row in rows)
    assert sum("train_mini" in row["image_path"] for row in rows) == 2

    paths = (tmp_path / "full_train_paths.txt").read_text(encoding="utf-8").splitlines()
    assert paths == [
        "train/00010_Species_A/c.jpg",
        "train/00010_Species_A/d.jpg",
        "train/00020_Species_B/g.jpg",
    ]
    assert config["raw_selected_images"] == 7
    assert config["excluded_validation_images"] == 2
    assert config["training_images"] == 5
    assert config["reused_train_mini_images"] == 2
    assert config["additional_images_to_extract"] == 3


def test_build_rejects_missing_validation_image(tmp_path: Path) -> None:
    fixture = prepare_fixture(tmp_path)
    payload = json.loads(fixture["json"].read_text(encoding="utf-8"))
    payload["images"] = [
        image for image in payload["images"] if not image["file_name"].endswith("/b.jpg")
    ]
    payload["annotations"] = [
        annotation for annotation in payload["annotations"] if annotation["image_id"] != 101
    ]
    fixture["json"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not fully represented"):
        run_build(tmp_path, fixture)
