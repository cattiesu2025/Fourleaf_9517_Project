#!/usr/bin/env python3
"""Copy images referenced by split metadata into a compact dataset directory."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from src.common.cli import ArgumentParser


@dataclass(frozen=True)
class CopySummary:
    referenced_images: int
    available_images: int
    copied_images: int
    total_bytes: int
    missing_paths: tuple[Path, ...]


def collect_referenced_paths(
    metadata_dir: Path,
    splits: Iterable[str] = ("train", "val", "test"),
) -> set[PurePosixPath]:
    """Read unique, safe image paths from the requested split CSV files."""
    referenced: set[PurePosixPath] = set()
    for split in splits:
        csv_path = metadata_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Split metadata not found: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "image_path" not in (reader.fieldnames or ()):
                raise ValueError(f"{csv_path} is missing the image_path column")
            for row in reader:
                image_path = PurePosixPath(row["image_path"].replace("\\", "/"))
                if image_path.is_absolute() or ".." in image_path.parts:
                    raise ValueError(f"Unsafe image path in {csv_path}: {image_path}")
                referenced.add(image_path)
    return referenced


def _destination_relative_path(image_path: PurePosixPath) -> PurePosixPath:
    raw_prefix = ("data", "raw")
    if image_path.parts[: len(raw_prefix)] == raw_prefix:
        relative_parts = image_path.parts[len(raw_prefix) :]
        if not relative_parts:
            raise ValueError(f"Image path does not identify a file: {image_path}")
        return PurePosixPath(*relative_parts)
    return image_path


def copy_selected_images(
    *,
    metadata_dir: Path,
    source_root: Path,
    output_dir: Path,
    splits: Iterable[str] = ("train", "val", "test"),
    dry_run: bool = False,
    overwrite_existing: bool = False,
    allow_missing: bool = False,
) -> CopySummary:
    """Copy referenced images while preserving paths below ``data/raw``."""
    referenced = collect_referenced_paths(metadata_dir, splits)
    resolved_source_root = source_root.resolve()
    resolved_output_dir = output_dir.resolve()
    planned: list[tuple[Path, Path, int]] = []
    missing: list[Path] = []
    destinations: set[Path] = set()

    for image_path in sorted(referenced, key=str):
        source = (source_root / Path(*image_path.parts)).resolve()
        try:
            source.relative_to(resolved_source_root)
        except ValueError as exc:
            raise ValueError(f"Image path escapes source root: {image_path}") from exc

        if not source.is_file():
            missing.append(source)
            continue

        relative_destination = _destination_relative_path(image_path)
        destination = output_dir / Path(*relative_destination.parts)
        resolved_destination = destination.resolve()
        try:
            resolved_destination.relative_to(resolved_output_dir)
        except ValueError as exc:
            raise ValueError(f"Image path escapes output directory: {image_path}") from exc
        if resolved_destination in destinations:
            raise ValueError(f"Multiple source paths map to {destination}")
        destinations.add(resolved_destination)
        planned.append((source, destination, source.stat().st_size))

    if missing and not allow_missing:
        examples = "\n  ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} referenced images are missing; examples:\n  {examples}"
        )

    existing_destinations = [destination for _, destination, _ in planned if destination.exists()]
    if existing_destinations and not dry_run and not overwrite_existing:
        examples = "\n  ".join(str(path) for path in existing_destinations[:5])
        raise FileExistsError(
            f"{len(existing_destinations)} destination files already exist; examples:\n"
            f"  {examples}\nPass --overwrite-existing to replace them."
        )

    copied = 0
    if not dry_run:
        for source, destination, _ in planned:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1

    return CopySummary(
        referenced_images=len(referenced),
        available_images=len(planned),
        copied_images=copied,
        total_bytes=sum(size for _, _, size in planned),
        missing_paths=tuple(missing),
    )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--source_root", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("data/subset"))
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--allow_missing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = copy_selected_images(
        metadata_dir=args.metadata_dir,
        source_root=args.source_root,
        output_dir=args.output_dir,
        splits=args.splits,
        dry_run=args.dry_run,
        overwrite_existing=args.overwrite_existing,
        allow_missing=args.allow_missing,
    )
    action = "Would copy" if args.dry_run else "Copied"
    size_gib = summary.total_bytes / 1024**3
    print(
        f"{action} {summary.available_images}/{summary.referenced_images} images "
        f"({size_gib:.2f} GiB) to {args.output_dir}."
    )
    if summary.missing_paths:
        print(f"Skipped {len(summary.missing_paths)} missing images.")


if __name__ == "__main__":
    main()
