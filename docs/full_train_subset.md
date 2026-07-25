# iNaturalist 2021 Full-Train Subset

This is an optional expansion of the existing fixed 500-class experiment. It
does not overwrite `train.csv`, `val.csv`, or `test.csv`.

## What stays fixed

- The existing 500 classes and `class_idx = 0..499` mapping are preserved.
- The existing 5,000-image validation split remains unchanged and is excluded
  from full-train metadata.
- The existing 5,000-image official-validation test split remains unchanged.
- The existing 20,000 `train_mini` training images are reused.

The generated `train_full.csv` adds every other full-train image belonging to
the selected classes. `full_train_paths.txt` contains only the additional
images that are not already present in `train_mini`.

The candidate metadata is intentionally generated before pixel-level quality
control. After extraction, a separate scan creates `train_full_clean.csv`
without overwriting the candidate CSV.

## Katana preparation

Run large transfers and archive extraction on Katana Data Mover (KDM):

```bash
ssh z5535967@kdm.restech.unsw.edu.au
cd /srv/scratch/z5535967/Fourleaf_9517_Project
```

Install the updated requirements in the project environment, then download the
221 MB compressed annotation archive. This produces an approximately 1 GB
`data/raw/train.json` file:

```bash
python -m pip install -r requirements.txt

curl --fail --location \
  https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz \
  | tar -xzf - -C data/raw
```

Generate and inspect the metadata before starting the image transfer:

```bash
python scripts/build_full_train_subset.py

wc -l \
  data/metadata/train_full.csv \
  data/metadata/full_train_paths.txt
cat data/metadata/split_config_full.json
```

For the current selected classes, the expected counts are:

- 134,549 selected images in full train before holdout exclusion
- 5,000 existing validation images excluded
- 129,549 training rows
- 20,000 existing `train_mini` images reused
- 109,549 additional images to extract

## Extract the additional images

The official S3 bucket exposes one `train.tar.gz` object rather than individual
image objects. Selective extraction therefore avoids storing the 240 GB
archive, but the complete archive still passes through the network stream.

```bash
tmux new -s inat-full

bash scripts/extract_full_train_subset.sh \
  --confirm-240gb-download
```

Do not upload the extracted images to Git or Google Drive. Keep them in
`/srv/scratch/z5535967`; scratch is not backed up.

## Scan image quality and create the clean CSV

Run the quality scan only after all requested images have been extracted:

```bash
python scripts/scan_dataset_quality.py
```

The scan reads every image referenced by `train_full.csv`, `val.csv`, and
`test.csv` and checks:

- missing or corrupt image files;
- exact file duplicates within training and between training and holdout sets;
- perceptually similar images using a 64-bit perceptual hash;
- blur using variance of the Laplacian;
- extremely dark or overexposed images using grayscale intensity statistics.

The default removal policy is deliberately conservative:

- remove corrupt training images;
- remove exact training duplicates, retaining one deterministic copy when all
  labels agree;
- remove every training copy that is byte-identical to a validation or test
  image;
- remove all members of an exact-duplicate group if their class labels
  conflict;
- flag near duplicates, blur, darkness, and overexposure for review, but do
  not remove them automatically.

Generated files:

```text
data/metadata/train_full_clean.csv
data/metadata/data_quality_report.csv
data/metadata/data_quality_summary.json
```

Review the summary before training:

```bash
cat data/metadata/data_quality_summary.json
```

The implementation was also verified against the current 30,000-image
mini-based split. All files decoded successfully. With the default thresholds,
the scan flagged 218 blurry, 44 extremely dark, 32 overexposed, and 97
near-duplicate candidates across train/validation/test. It found one
byte-identical train-validation pair with the same class label and excluded the
training copy. These numbers validate the scanner only; the full-subset results
must be generated again after extracting all additional images.

After quality control, use the clean training CSV without changing validation:

```bash
python src/transfer/train.py \
  --train_csv data/metadata/train_full_clean.csv \
  --val_csv data/metadata/val.csv \
  [other existing arguments]
```
