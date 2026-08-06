#!/usr/bin/env python3
"""Build the fixed iNaturalist class split and metadata files."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from src.common.cli import ArgumentParser

CSV_COLUMNS = (
    "image_id",
    "image_path",
    "original_class_id",
    "class_idx",
    "class_name",
    "split",
)
SELECTED_CLASS_COLUMNS = (
    "class_idx",
    "original_class_id",
    "class_name",
    "category",
)


@dataclass(frozen=True)
class AnnotationData:
    images: dict[int, str]
    category_by_image: dict[int, int]
    categories: dict[int, dict[str, str]]


def load_annotation_file(path: Path) -> AnnotationData:
    """Load the COCO-style image, annotation, and category records."""
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    required_sections = {"images", "annotations", "categories"}
    missing_sections = sorted(required_sections - set(payload))
    if missing_sections:
        raise ValueError(f"{path} is missing sections: {missing_sections}")

    images: dict[int, str] = {}
    for image in payload["images"]:
        image_id = int(image["id"])
        if image_id in images:
            raise ValueError(f"Duplicate image id {image_id} in {path}")
        images[image_id] = str(image["file_name"])

    category_by_image: dict[int, int] = {}
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        previous = category_by_image.setdefault(image_id, category_id)
        if previous != category_id:
            raise ValueError(f"Image {image_id} has conflicting categories in {path}")

    categories: dict[int, dict[str, str]] = {}
    for category in payload["categories"]:
        category_id = int(category["id"])
        if category_id in categories:
            raise ValueError(f"Duplicate category id {category_id} in {path}")
        categories[category_id] = {
            "name": str(category.get("name", f"category_{category_id}")),
            "category": str(category.get("supercategory") or category.get("kingdom") or "unknown"),
        }

    unknown_categories = set(category_by_image.values()) - set(categories)
    if unknown_categories:
        examples = sorted(unknown_categories)[:5]
        raise ValueError(f"{path} annotations reference unknown categories: {examples}")
    return AnnotationData(images, category_by_image, categories)


def group_images_by_category(data: AnnotationData) -> dict[int, list[int]]:
    """Group annotated image identifiers by category."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for image_id, category_id in data.category_by_image.items():
        if image_id in data.images:
            grouped[category_id].append(image_id)
    return dict(grouped)


def normalize_image_path(raw_root: Path, file_name: str) -> str:
    """Return a portable path below the configured raw-data directory."""
    normalized = PurePosixPath(file_name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Unsafe image path in annotation data: {file_name}")

    root = PurePosixPath(raw_root.as_posix())
    if normalized.parts[: len(root.parts)] == root.parts:
        return normalized.as_posix()
    return (root / normalized).as_posix()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _ensure_outputs_available(output_dir: Path, names: Iterable[str], force: bool) -> None:
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not force:
        paths = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to replace existing metadata. Pass --force after verifying the "
            f"fixed split may be rebuilt:\n  {paths}"
        )


def build_dataset(
    *,
    train_json: Path,
    validation_json: Path,
    output_dir: Path,
    raw_root: Path,
    num_classes: int,
    train_per_class: int,
    validation_per_class: int,
    test_per_class: int,
    seed: int,
    class_selection_method: str = "uniform_random_sampling",
    force: bool = False,
) -> dict[str, Any]:
    """Select classes and write the fixed train, validation, and test metadata."""
    requested_counts = {
        "num_classes": num_classes,
        "train_per_class": train_per_class,
        "validation_per_class": validation_per_class,
        "test_per_class": test_per_class,
    }
    invalid = {name: value for name, value in requested_counts.items() if value <= 0}
    if invalid:
        raise ValueError(f"Split sizes must be positive: {invalid}")
    if class_selection_method != "uniform_random_sampling":
        raise ValueError("class_selection_method must be 'uniform_random_sampling'")

    output_names = (
        "selected_classes.csv",
        "class_to_idx.json",
        "idx_to_class.json",
        "train.csv",
        "val.csv",
        "test.csv",
        "split_config.json",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_outputs_available(output_dir, output_names, force)

    train_data = load_annotation_file(train_json)
    validation_data = load_annotation_file(validation_json)
    train_groups = group_images_by_category(train_data)
    validation_groups = group_images_by_category(validation_data)

    minimum_training_images = train_per_class + validation_per_class
    eligible_categories = sorted(
        category_id
        for category_id in train_data.categories
        if len(train_groups.get(category_id, ())) >= minimum_training_images
        and len(validation_groups.get(category_id, ())) >= test_per_class
    )
    if len(eligible_categories) < num_classes:
        raise ValueError(
            f"Only {len(eligible_categories)} eligible categories are available; "
            f"{num_classes} are required."
        )

    random_state = random.Random(seed)
    selected_categories = sorted(random_state.sample(eligible_categories, num_classes))
    random_state.shuffle(selected_categories)
    class_to_idx = {
        str(category_id): class_idx for class_idx, category_id in enumerate(selected_categories)
    }
    idx_to_class = {
        str(class_idx): {
            "original_class_id": category_id,
            "class_name": train_data.categories[category_id]["name"],
            "category": train_data.categories[category_id]["category"],
        }
        for class_idx, category_id in enumerate(selected_categories)
    }

    selected_rows: list[dict[str, Any]] = []
    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    next_image_id = 0
    for class_idx, category_id in enumerate(selected_categories):
        category = train_data.categories[category_id]
        selected_rows.append(
            {
                "class_idx": class_idx,
                "original_class_id": category_id,
                "class_name": category["name"],
                "category": category["category"],
            }
        )

        training_pool = list(train_groups[category_id])
        test_pool = list(validation_groups[category_id])
        random_state.shuffle(training_pool)
        random_state.shuffle(test_pool)
        selections = (
            ("train", training_pool[:train_per_class], train_data),
            (
                "val",
                training_pool[train_per_class : train_per_class + validation_per_class],
                train_data,
            ),
            ("test", test_pool[:test_per_class], validation_data),
        )
        for split_name, image_ids, source in selections:
            for source_image_id in image_ids:
                split_rows[split_name].append(
                    {
                        "image_id": next_image_id,
                        "image_path": normalize_image_path(
                            raw_root, source.images[source_image_id]
                        ),
                        "original_class_id": category_id,
                        "class_idx": class_idx,
                        "class_name": category["name"],
                        "split": split_name,
                    }
                )
                next_image_id += 1

    _atomic_write_csv(output_dir / "selected_classes.csv", SELECTED_CLASS_COLUMNS, selected_rows)
    _atomic_write_json(output_dir / "class_to_idx.json", class_to_idx)
    _atomic_write_json(output_dir / "idx_to_class.json", idx_to_class)
    for split_name, rows in split_rows.items():
        _atomic_write_csv(output_dir / f"{split_name}.csv", CSV_COLUMNS, rows)

    config = {
        "random_seed": seed,
        "num_classes": num_classes,
        "class_selection_method": class_selection_method,
        "train_images_per_class": train_per_class,
        "val_images_per_class": validation_per_class,
        "test_images_per_class": test_per_class,
        "image_size": 224,
        "dataset": "iNaturalist-2021",
    }
    _atomic_write_json(output_dir / "split_config.json", config)
    return config


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--train_json", type=Path, default=Path("data/raw/train_mini.json"))
    parser.add_argument(
        "--validation_json",
        "--val_json",
        dest="validation_json",
        type=Path,
        default=Path("data/raw/val.json"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--raw_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--num_classes", type=int, default=500)
    parser.add_argument("--train_per_class", type=int, default=40)
    parser.add_argument(
        "--validation_per_class",
        "--val_per_class",
        dest="validation_per_class",
        type=int,
        default=10,
    )
    parser.add_argument("--test_per_class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--class_selection_method", default="uniform_random_sampling")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = build_dataset(
        train_json=args.train_json,
        validation_json=args.validation_json,
        output_dir=args.output_dir,
        raw_root=args.raw_root,
        num_classes=args.num_classes,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
        class_selection_method=args.class_selection_method,
        force=args.force,
    )
    print(
        f"Wrote {config['num_classes']} classes to {args.output_dir} "
        f"with seed {config['random_seed']}."
    )


if __name__ == "__main__":
    main()
