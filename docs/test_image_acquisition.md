# Official test-image acquisition record

The project's 5,000-image test split is sampled from the labelled validation
split of iNaturalist 2021.  The image files are local data only and must not be
committed to Git.

## Sources

- Image archive: `https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz`
  - Content length: `8,931,661,582` bytes
  - Publisher metadata MD5: `f6f6e0e242e3d4c9569ba56400938afc`
- Official COCO-format metadata:
  `https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz`
  - MD5: `4d761e0f6a86cc63e8f7afc91f6a8f0b`
- Fixed project split: `data/metadata/test.csv`
  - SHA-256: `0c8c7a40a1c3870f7db20139f83cc07e35f2f0f6c75975d60c961cf4758b2281`

## Selection and validation

Only paths listed by `test.csv` are extracted.  The leading `data/raw/`
component is removed when matching archive members, and the selected members
are written back beneath `data/raw/val/`.

The split was cross-checked against the official `val.json` before extraction:

- 5,000 CSV rows and 5,000 unique image paths;
- all 5,000 paths occur in the official validation metadata;
- all 5,000 `original_class_id` values agree with the official annotations;
- no extra validation images are intentionally retained.

After extraction, verify that all 5,000 expected paths exist, that there are no
unexpected files under `data/raw/val/`, and that Pillow can decode and verify
every image.  The full archive checksum must be validated before the selected
members are accepted.

Local acquisition completed on 2026-07-26 with the following checks:

- reconstructed archive MD5: `f6f6e0e242e3d4c9569ba56400938afc` (match);
- expected files: 5,000; present files: 5,000;
- missing files: 0; unexpected files: 0;
- Pillow verification failures: 0; detected format: 5,000 JPEG;
- class directories: 500;
- retained image bytes: `452,765,582` (approximately 432 MiB).

## Why the full archive is read

The official image release is one gzip-compressed tar archive.  It does not
expose the competition filenames as individual S3 objects, and gzip does not
support reliable member-level HTTP range retrieval without a matching seek
index.  Consequently, obtaining this exact split requires downloading or
streaming the full compressed archive even though only the 5,000 listed members
are written to the project directory.
