#!/usr/bin/env python3
"""Scan iNaturalist images and create a conservatively cleaned train CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


SOURCE_COLUMNS = (
    "image_id",
    "image_path",
    "original_class_id",
    "class_idx",
    "class_name",
    "split",
)

REPORT_COLUMNS = SOURCE_COLUMNS + (
    "dataset_role",
    "file_status",
    "error",
    "file_size_bytes",
    "width",
    "height",
    "sha256",
    "phash",
    "laplacian_variance",
    "mean_brightness",
    "dark_pixel_fraction",
    "bright_pixel_fraction",
    "is_blurry",
    "is_too_dark",
    "is_overexposed",
    "exact_duplicate_scope",
    "exact_duplicate_of",
    "near_duplicate_scope",
    "near_duplicate_of",
    "near_duplicate_distance",
    "quality_flags",
    "exclude_from_clean",
    "exclusion_reason",
)


@dataclass(frozen=True)
class Thresholds:
    blur_laplacian_variance: float = 50.0
    dark_mean_brightness: float = 30.0
    dark_pixel_fraction: float = 0.50
    overexposed_mean_brightness: float = 225.0
    bright_pixel_fraction: float = 0.50
    near_duplicate_hamming_distance: int = 6


@dataclass(frozen=True)
class HashItem:
    key: str
    image_id: str
    role: str
    class_idx: int
    sha256: str
    phash: int


class HammingIndex:
    """Candidate index that guarantees recall within the configured radius."""

    def __init__(self, radius: int) -> None:
        if not 0 <= radius < 64:
            raise ValueError("Hamming radius must be in the range 0..63")
        self.radius = radius
        self.block_count = radius + 1
        base_size, remainder = divmod(64, self.block_count)
        self.block_sizes = [
            base_size + (1 if index < remainder else 0)
            for index in range(self.block_count)
        ]
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.items: list[HashItem] = []

    def _block_keys(self, value: int) -> Iterable[tuple[int, int]]:
        shift = 0
        for block_index, block_size in enumerate(self.block_sizes):
            mask = (1 << block_size) - 1
            yield block_index, (value >> shift) & mask
            shift += block_size

    def add(self, item: HashItem) -> None:
        item_index = len(self.items)
        self.items.append(item)
        for block_key in self._block_keys(item.phash):
            self.buckets[block_key].append(item_index)

    def nearest(self, item: HashItem) -> tuple[HashItem, int] | None:
        candidate_indices: set[int] = set()
        for block_key in self._block_keys(item.phash):
            candidate_indices.update(self.buckets.get(block_key, ()))

        best: tuple[HashItem, int] | None = None
        for candidate_index in candidate_indices:
            candidate = self.items[candidate_index]
            if candidate.sha256 == item.sha256:
                continue
            distance = (candidate.phash ^ item.phash).bit_count()
            if distance > self.radius:
                continue
            if best is None or (distance, candidate.key) < (best[1], best[0].key):
                best = candidate, distance
        return best


def read_source_csv(
    path: Path,
    role: str,
    max_images: int | None,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(SOURCE_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        rows = list(reader)

    if max_images is not None:
        rows = rows[:max_images]
    if not rows:
        raise ValueError(f"No rows selected from {path}")

    output: list[dict[str, str]] = []
    for row_index, row in enumerate(rows):
        record = {column: row[column] for column in SOURCE_COLUMNS}
        record["_role"] = role
        record["_row_index"] = str(row_index)
        record["_key"] = f"{role}:{row_index}:{row['image_id']}"
        output.append(record)
    return output


def perceptual_hash(grayscale: np.ndarray) -> int:
    resized = cv2.resize(grayscale, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))
    low_frequency = coefficients[:8, :8]
    median = float(np.median(low_frequency.ravel()[1:]))
    bits = low_frequency > median
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def scan_record(
    record: dict[str, str],
    thresholds: Thresholds,
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    path = Path(record["image_path"])
    result: dict[str, Any] = {
        "_key": record["_key"],
        "_role": record["_role"],
        "file_status": "ok",
        "error": "",
        "file_size_bytes": "",
        "width": "",
        "height": "",
        "sha256": "",
        "phash": "",
        "laplacian_variance": "",
        "mean_brightness": "",
        "dark_pixel_fraction": "",
        "bright_pixel_fraction": "",
        "is_blurry": False,
        "is_too_dark": False,
        "is_overexposed": False,
    }

    try:
        payload = path.read_bytes()
        result["file_size_bytes"] = len(payload)
        result["sha256"] = hashlib.sha256(payload).hexdigest()

        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.load()

        result["width"], result["height"] = rgb.size
        grayscale = np.asarray(rgb.convert("L"), dtype=np.uint8)
        quality_image = cv2.resize(
            grayscale,
            (256, 256),
            interpolation=cv2.INTER_AREA,
        )
        mean_brightness = float(quality_image.mean())
        dark_fraction = float(np.mean(quality_image <= 10))
        bright_fraction = float(np.mean(quality_image >= 245))
        laplacian_variance = float(
            cv2.Laplacian(quality_image, cv2.CV_64F).var()
        )
        phash_value = perceptual_hash(grayscale)

        result.update(
            {
                "phash": f"{phash_value:016x}",
                "laplacian_variance": laplacian_variance,
                "mean_brightness": mean_brightness,
                "dark_pixel_fraction": dark_fraction,
                "bright_pixel_fraction": bright_fraction,
                "is_blurry": (
                    laplacian_variance
                    < thresholds.blur_laplacian_variance
                ),
                "is_too_dark": (
                    mean_brightness <= thresholds.dark_mean_brightness
                    and dark_fraction >= thresholds.dark_pixel_fraction
                ),
                "is_overexposed": (
                    mean_brightness
                    >= thresholds.overexposed_mean_brightness
                    and bright_fraction >= thresholds.bright_pixel_fraction
                ),
            }
        )
    except FileNotFoundError:
        result["file_status"] = "missing"
        result["error"] = "FileNotFoundError"
    except (OSError, ValueError, SyntaxError, cv2.error) as exc:
        result["file_status"] = "corrupt"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def scan_all(
    records: list[dict[str, str]],
    thresholds: Thresholds,
    workers: int,
) -> list[dict[str, Any]]:
    worker = partial(scan_record, thresholds=thresholds)
    def serial_scan() -> list[dict[str, Any]]:
        return [
            worker(record)
            for record in tqdm(records, desc="Quality scan", unit="image")
        ]

    if workers == 1:
        return serial_scan()

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(worker, records, chunksize=32)
            return list(
                tqdm(
                    iterator,
                    total=len(records),
                    desc="Quality scan",
                    unit="image",
                )
            )
    except PermissionError as exc:
        print(
            "Parallel image scanning is unavailable "
            f"({type(exc).__name__}: {exc}); falling back to one worker."
        )
        return serial_scan()


def require_all_paths(records: list[dict[str, str]]) -> None:
    missing = [
        record["image_path"]
        for record in records
        if not Path(record["image_path"]).is_file()
    ]
    if missing:
        examples = "\n".join(f"  {path}" for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing):,} referenced images are missing. Complete extraction "
            f"before quality scanning. Examples:\n{examples}"
        )


def add_reason(
    reasons: dict[str, list[str]],
    key: str,
    reason: str,
) -> None:
    if reason not in reasons[key]:
        reasons[key].append(reason)


def exact_duplicate_analysis(
    records_by_key: dict[str, dict[str, str]],
    scans_by_key: dict[str, dict[str, Any]],
    reasons: dict[str, list[str]],
) -> dict[str, dict[str, str]]:
    by_digest: dict[str, list[str]] = defaultdict(list)
    annotations: dict[str, dict[str, str]] = defaultdict(dict)
    for key, scan in scans_by_key.items():
        if scan["file_status"] == "ok":
            by_digest[str(scan["sha256"])].append(key)

    for keys in by_digest.values():
        if len(keys) < 2:
            continue
        train_keys = [
            key for key in keys if records_by_key[key]["_role"] == "train"
        ]
        holdout_keys = [
            key for key in keys if records_by_key[key]["_role"] != "train"
        ]

        if holdout_keys:
            reference = min(
                holdout_keys,
                key=lambda key: (
                    records_by_key[key]["_role"],
                    records_by_key[key]["image_id"],
                ),
            )
            reference_label = (
                f"{records_by_key[reference]['_role']}:"
                f"{records_by_key[reference]['image_id']}"
            )
            for key in train_keys:
                annotations[key] = {
                    "scope": "holdout",
                    "of": reference_label,
                }
                add_reason(reasons, key, "exact_duplicate_holdout")
            continue

        if len(train_keys) < 2:
            continue

        labels = {
            int(records_by_key[key]["class_idx"])
            for key in train_keys
        }
        ordered = sorted(
            train_keys,
            key=lambda key: (
                int(records_by_key[key]["_row_index"]),
            ),
        )
        if len(labels) > 1:
            for key in ordered:
                annotations[key] = {
                    "scope": "train_label_conflict",
                    "of": "",
                }
                add_reason(reasons, key, "exact_duplicate_label_conflict")
        else:
            keeper = ordered[0]
            keeper_label = f"train:{records_by_key[keeper]['image_id']}"
            for key in ordered[1:]:
                annotations[key] = {
                    "scope": "train",
                    "of": keeper_label,
                }
                add_reason(reasons, key, "exact_duplicate_train")
    return annotations


def to_hash_item(
    key: str,
    records_by_key: dict[str, dict[str, str]],
    scans_by_key: dict[str, dict[str, Any]],
) -> HashItem:
    record = records_by_key[key]
    scan = scans_by_key[key]
    return HashItem(
        key=key,
        image_id=str(record["image_id"]),
        role=record["_role"],
        class_idx=int(record["class_idx"]),
        sha256=str(scan["sha256"]),
        phash=int(str(scan["phash"]), 16),
    )


def near_duplicate_analysis(
    records_by_key: dict[str, dict[str, str]],
    scans_by_key: dict[str, dict[str, Any]],
    excluded_reasons: dict[str, list[str]],
    radius: int,
) -> dict[str, dict[str, str | int]]:
    annotations: dict[str, dict[str, str | int]] = {}
    valid_keys = [
        key
        for key, scan in scans_by_key.items()
        if scan["file_status"] == "ok"
    ]
    holdout_keys = sorted(
        (
            key
            for key in valid_keys
            if records_by_key[key]["_role"] != "train"
        ),
        key=lambda key: (
            records_by_key[key]["_role"],
            records_by_key[key]["image_id"],
        ),
    )
    train_keys = sorted(
        (
            key
            for key in valid_keys
            if records_by_key[key]["_role"] == "train"
        ),
        key=lambda key: (
            int(records_by_key[key]["class_idx"]),
            records_by_key[key]["image_path"],
            records_by_key[key]["image_id"],
        ),
    )

    holdout_index = HammingIndex(radius)
    for key in holdout_keys:
        holdout_index.add(to_hash_item(key, records_by_key, scans_by_key))

    train_index = HammingIndex(radius)
    for key in train_keys:
        item = to_hash_item(key, records_by_key, scans_by_key)
        holdout_match = holdout_index.nearest(item)
        if holdout_match is not None:
            match, distance = holdout_match
            annotations[key] = {
                "scope": match.role,
                "of": f"{match.role}:{match.image_id}",
                "distance": distance,
                "cross_label": int(match.class_idx != item.class_idx),
            }
        else:
            train_match = train_index.nearest(item)
            if train_match is not None:
                match, distance = train_match
                annotations[key] = {
                    "scope": "train",
                    "of": f"train:{match.image_id}",
                    "distance": distance,
                    "cross_label": int(match.class_idx != item.class_idx),
                }

        if not excluded_reasons.get(key):
            train_index.add(item)
    return annotations


def atomic_write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_outputs(
    *,
    records: list[dict[str, str]],
    scans: list[dict[str, Any]],
    thresholds: Thresholds,
    clean_csv: Path,
    report_csv: Path,
    summary_json: Path,
    partial_scan: bool,
) -> dict[str, Any]:
    records_by_key = {record["_key"]: record for record in records}
    scans_by_key = {scan["_key"]: scan for scan in scans}
    if set(records_by_key) != set(scans_by_key):
        raise ValueError("Scan results do not align with metadata rows")

    reasons: dict[str, list[str]] = defaultdict(list)
    for key, scan in scans_by_key.items():
        if (
            records_by_key[key]["_role"] == "train"
            and scan["file_status"] != "ok"
        ):
            add_reason(reasons, key, scan["file_status"])

    exact_annotations = exact_duplicate_analysis(
        records_by_key,
        scans_by_key,
        reasons,
    )
    near_annotations = near_duplicate_analysis(
        records_by_key,
        scans_by_key,
        reasons,
        thresholds.near_duplicate_hamming_distance,
    )

    report_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, str]] = []
    flag_counts: Counter[str] = Counter()
    flag_counts_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    exclusion_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for record in records:
        key = record["_key"]
        role = record["_role"]
        scan = scans_by_key[key]
        flags: list[str] = []
        if scan["file_status"] != "ok":
            flags.append(str(scan["file_status"]))
        if scan["is_blurry"]:
            flags.append("blurry")
        if scan["is_too_dark"]:
            flags.append("too_dark")
        if scan["is_overexposed"]:
            flags.append("overexposed")

        exact = exact_annotations.get(key, {})
        if exact:
            flags.append(f"exact_duplicate_{exact['scope']}")

        near = near_annotations.get(key, {})
        if near:
            flags.append(f"near_duplicate_{near['scope']}")
            if int(near["cross_label"]):
                flags.append("near_duplicate_cross_label")

        exclusion_reasons = reasons.get(key, [])
        for flag in flags:
            flag_counts[flag] += 1
            flag_counts_by_role[role][flag] += 1
        for reason in exclusion_reasons:
            exclusion_counts[reason] += 1
        role_counts[role] += 1
        status_counts[f"{role}:{scan['file_status']}"] += 1

        report_row = {column: record[column] for column in SOURCE_COLUMNS}
        report_row.update(
            {
                "dataset_role": role,
                "file_status": scan["file_status"],
                "error": scan["error"],
                "file_size_bytes": scan["file_size_bytes"],
                "width": scan["width"],
                "height": scan["height"],
                "sha256": scan["sha256"],
                "phash": scan["phash"],
                "laplacian_variance": scan["laplacian_variance"],
                "mean_brightness": scan["mean_brightness"],
                "dark_pixel_fraction": scan["dark_pixel_fraction"],
                "bright_pixel_fraction": scan["bright_pixel_fraction"],
                "is_blurry": scan["is_blurry"],
                "is_too_dark": scan["is_too_dark"],
                "is_overexposed": scan["is_overexposed"],
                "exact_duplicate_scope": exact.get("scope", ""),
                "exact_duplicate_of": exact.get("of", ""),
                "near_duplicate_scope": near.get("scope", ""),
                "near_duplicate_of": near.get("of", ""),
                "near_duplicate_distance": near.get("distance", ""),
                "quality_flags": ";".join(flags),
                "exclude_from_clean": bool(exclusion_reasons),
                "exclusion_reason": ";".join(exclusion_reasons),
            }
        )
        report_rows.append(report_row)

        if role == "train" and not exclusion_reasons:
            clean_rows.append(
                {column: record[column] for column in SOURCE_COLUMNS}
            )

    train_input_count = role_counts["train"]
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "partial_scan": partial_scan,
        "input_images": dict(sorted(role_counts.items())),
        "file_status_counts": dict(sorted(status_counts.items())),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "quality_flag_counts_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(flag_counts_by_role.items())
        },
        "train_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "train_input_images": train_input_count,
        "train_clean_images": len(clean_rows),
        "train_excluded_images": train_input_count - len(clean_rows),
        "thresholds": {
            "blur_laplacian_variance_below": (
                thresholds.blur_laplacian_variance
            ),
            "dark_mean_brightness_at_most": (
                thresholds.dark_mean_brightness
            ),
            "dark_pixel_fraction_at_least": (
                thresholds.dark_pixel_fraction
            ),
            "overexposed_mean_brightness_at_least": (
                thresholds.overexposed_mean_brightness
            ),
            "bright_pixel_fraction_at_least": (
                thresholds.bright_pixel_fraction
            ),
            "near_duplicate_phash_hamming_distance_at_most": (
                thresholds.near_duplicate_hamming_distance
            ),
        },
        "removal_policy": {
            "corrupt_or_missing_train_image": "exclude",
            "exact_duplicate_of_validation_or_test": "exclude",
            "exact_duplicate_within_train_same_label": "keep_one",
            "exact_duplicate_within_train_conflicting_labels": "exclude_all",
            "near_duplicate": "flag_only",
            "blurry": "flag_only",
            "too_dark": "flag_only",
            "overexposed": "flag_only",
        },
    }

    atomic_write_csv(report_csv, REPORT_COLUMNS, report_rows)
    atomic_write_csv(clean_csv, SOURCE_COLUMNS, clean_rows)
    atomic_write_json(summary_json, summary)
    return summary


def scan_dataset_quality(
    *,
    train_csv: Path,
    validation_csv: Path,
    test_csv: Path,
    clean_csv: Path,
    report_csv: Path,
    summary_json: Path,
    thresholds: Thresholds,
    workers: int,
    max_images: int | None = None,
) -> dict[str, Any]:
    records = (
        read_source_csv(train_csv, "train", max_images)
        + read_source_csv(validation_csv, "validation", max_images)
        + read_source_csv(test_csv, "test", max_images)
    )
    require_all_paths(records)
    scans = scan_all(records, thresholds, workers)
    return build_outputs(
        records=records,
        scans=scans,
        thresholds=thresholds,
        clean_csv=clean_csv,
        report_csv=report_csv,
        summary_json=summary_json,
        partial_scan=max_images is not None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan train/validation/test images for corruption, duplicates, "
            "blur, darkness, and overexposure."
        )
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/metadata/train_full.csv"),
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=Path("data/metadata/val.csv"),
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/metadata/test.csv"),
    )
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=Path("data/metadata/train_full_clean.csv"),
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("data/metadata/data_quality_report.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/metadata/data_quality_summary.json"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--blur-threshold", type=float, default=50.0)
    parser.add_argument("--dark-mean-threshold", type=float, default=30.0)
    parser.add_argument("--dark-fraction-threshold", type=float, default=0.50)
    parser.add_argument(
        "--overexposed-mean-threshold",
        type=float,
        default=225.0,
    )
    parser.add_argument("--bright-fraction-threshold", type=float, default=0.50)
    parser.add_argument("--near-duplicate-distance", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.max_images is not None and args.max_images < 1:
        raise SystemExit("--max-images must be at least 1")

    thresholds = Thresholds(
        blur_laplacian_variance=args.blur_threshold,
        dark_mean_brightness=args.dark_mean_threshold,
        dark_pixel_fraction=args.dark_fraction_threshold,
        overexposed_mean_brightness=args.overexposed_mean_threshold,
        bright_pixel_fraction=args.bright_fraction_threshold,
        near_duplicate_hamming_distance=args.near_duplicate_distance,
    )
    summary = scan_dataset_quality(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        test_csv=args.test_csv,
        clean_csv=args.clean_csv,
        report_csv=args.report_csv,
        summary_json=args.summary_json,
        thresholds=thresholds,
        workers=args.workers,
        max_images=args.max_images,
    )

    print("\nDataset quality scan completed:")
    print(f"  train input:     {summary['train_input_images']:,}")
    print(f"  train clean:     {summary['train_clean_images']:,}")
    print(f"  train excluded:  {summary['train_excluded_images']:,}")
    print(f"  report:          {args.report_csv}")
    print(f"  summary:         {args.summary_json}")


if __name__ == "__main__":
    main()
