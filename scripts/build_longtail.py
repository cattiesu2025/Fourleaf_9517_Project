#!/usr/bin/env python3
"""Build deterministic long-tail training metadata from the fixed split."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
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


def read_training_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training metadata not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(CSV_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Training metadata is empty: {path}")
    return rows


def exponential_decay_counts(
    num_classes: int,
    minimum_per_class: int,
    maximum_per_class: int,
) -> list[int]:
    """Return target counts that decay exponentially from head to tail."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if minimum_per_class <= 0 or maximum_per_class < minimum_per_class:
        raise ValueError("minimum_per_class must be positive and no greater than maximum_per_class")
    if num_classes == 1:
        return [maximum_per_class]

    decay = (minimum_per_class / maximum_per_class) ** (1.0 / (num_classes - 1))
    return [
        max(minimum_per_class, round(maximum_per_class * decay**rank))
        for rank in range(num_classes)
    ]


def _atomic_write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _refuse_existing(paths: Iterable[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing long-tail metadata:\n  {joined}\n"
            "Pass --force to rebuild it."
        )


def build_longtail_metadata(
    *,
    train_csv: Path,
    output_dir: Path,
    seed: int,
    minimum_per_class: int,
    maximum_per_class: int,
    static_oversample: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Write a long-tail subset and optional statically balanced metadata."""
    rows = read_training_rows(train_csv)
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["class_idx"])].append(row)

    class_indices = sorted(grouped)
    target_counts = exponential_decay_counts(
        len(class_indices), minimum_per_class, maximum_per_class
    )
    too_small = {
        class_idx: len(grouped[class_idx])
        for class_idx in class_indices
        if len(grouped[class_idx]) < minimum_per_class
    }
    if too_small:
        examples = dict(list(too_small.items())[:5])
        raise ValueError(
            f"Every class must contain at least minimum_per_class rows; examples: {examples}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    longtail_path = output_dir / "longtail_train.csv"
    config_path = output_dir / "longtail_config.json"
    resampled_path = output_dir / "longtail_resampled_train.csv"
    paths = [longtail_path, config_path]
    if static_oversample:
        paths.append(resampled_path)
    _refuse_existing(paths, force)

    random_state = random.Random(seed)
    class_order = list(class_indices)
    random_state.shuffle(class_order)

    longtail_rows: list[dict[str, str]] = []
    retained_by_class: dict[int, list[dict[str, str]]] = {}
    for rank, class_idx in enumerate(class_order):
        pool = list(grouped[class_idx])
        random_state.shuffle(pool)
        retained = pool[: min(target_counts[rank], len(pool))]
        retained_by_class[class_idx] = retained
        longtail_rows.extend(retained)
    _atomic_write_csv(longtail_path, longtail_rows)

    resampling_strategy = "weighted_random_sampler"
    if static_oversample:
        resampled_rows: list[dict[str, str]] = []
        for class_idx in class_order:
            retained = retained_by_class[class_idx]
            resampled_rows.extend(
                retained[index % len(retained)] for index in range(maximum_per_class)
            )
        _atomic_write_csv(resampled_path, resampled_rows)
        resampling_strategy = "static_oversampling"

    retained_counts = [len(rows_for_class) for rows_for_class in retained_by_class.values()]
    config = {
        "random_seed": seed,
        "longtail_ratio": "exponential_decay",
        "min_images_per_class": minimum_per_class,
        "max_images_per_class": maximum_per_class,
        "resampling_strategy": resampling_strategy,
        "num_classes": len(class_indices),
        "num_images": len(longtail_rows),
        "observed_min_images_per_class": min(retained_counts),
        "observed_max_images_per_class": max(retained_counts),
    }
    _atomic_write_json(config_path, config)
    return config


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", type=Path, default=Path("data/metadata/train.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument(
        "--minimum_per_class",
        "--min_per_class",
        dest="minimum_per_class",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--maximum_per_class",
        "--max_per_class",
        dest="maximum_per_class",
        type=int,
        default=40,
    )
    parser.add_argument("--static_oversample", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = build_longtail_metadata(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        minimum_per_class=args.minimum_per_class,
        maximum_per_class=args.maximum_per_class,
        static_oversample=args.static_oversample,
        force=args.force,
    )
    print(
        f"Wrote {config['num_images']} long-tail rows across "
        f"{config['num_classes']} classes to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
