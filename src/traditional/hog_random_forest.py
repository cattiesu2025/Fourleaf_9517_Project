"""
hog_random_forest.py
---------------------
Method 2: HOG + Random Forest

Example usage:

  # Quick test on a small subset (spec section 13.1: check feature dimension,
  # runtime, and memory usage on 50-100 classes before running on the full set)
  python -m src.traditional.hog_random_forest \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/val.csv \
      --num_classes 50 \
      --output_dir outputs/traditional/hog_random_forest_smoketest

  # Full run
  python -m src.traditional.hog_random_forest \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/test.csv \
      --output_dir outputs/traditional/hog_random_forest

  # Robustness
  python -m src.traditional.hog_random_forest \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/test.csv \
      --load_model outputs/traditional/hog_random_forest/rf_model.pkl \
      --degradation blur --severity 2 \
      --output_dir outputs/robustness/hog_random_forest/blur/severity_2

  # Hyperparameter sweep with feature caching: the first run extracts and
  # caches HOG features (slow, ~tens of minutes for 20000 images); every
  # subsequent run with a different --n_estimators / --max_depth reuses the
  # cached features and only re-runs RandomForest training (fast).
  python -m src.traditional.hog_random_forest \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/test.csv \
      --cache_dir outputs/traditional/hog_random_forest/feature_cache \
      --n_estimators 300 \
      --output_dir outputs/traditional/hog_random_forest_run1

  python -m src.traditional.hog_random_forest \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/test.csv \
      --cache_dir outputs/traditional/hog_random_forest/feature_cache \
      --n_estimators 500 --max_depth 30 \
      --output_dir outputs/traditional/hog_random_forest_run2

  # Finer HOG cells (more detail, bigger feature vector, slower). Note this
  # changes the cache key (see _cache_filename), so it triggers fresh
  # extraction the first time, then caches normally.
  python -m src.traditional.hog_random_forest \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/val.csv \
      --pixels_per_cell 8 \
      --cache_dir outputs/traditional/hog_random_forest/feature_cache \
      --output_dir outputs/traditional/hog_random_forest_cell8
"""

import os
import argparse
import pickle

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier

from src.traditional.common_io import (
    INatDataset, Timer,
    save_predictions_csv, save_scores_npz, save_runtime_json,
    load_degradation_fn, degrade_pil_image,
)

METHOD_NAME = "hog_random_forest"

HOG_ORIENTATIONS = 9
HOG_CELLS_PER_BLOCK = (2, 2)
# Default image size / cell size -- both are now CLI-tunable via --image_size
# and --pixels_per_cell, since finer cells often help fine-grained species
# classification at the cost of a larger feature vector and slower training.
DEFAULT_HOG_IMAGE_SIZE = 128
DEFAULT_HOG_PIXELS_PER_CELL = 16


def extract_hog_feature(gray_img: np.ndarray, image_size: int, pixels_per_cell: int) -> np.ndarray:
    resized = cv2.resize(gray_img, (image_size, image_size))
    feat = hog(
        resized,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def dataset_to_hog_features(dataset: INatDataset, image_size: int, pixels_per_cell: int,
                             degrade_fn=None, degradation_type=None, severity=None, seed=9517):
    feats = []
    for i, rec in enumerate(dataset):
        img = dataset.load_image(rec, as_gray=False)
        if degrade_fn is not None:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            pil_img = degrade_pil_image(degrade_fn, pil_img, degradation_type, severity, seed=seed)
            img = np.array(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feats.append(extract_hog_feature(gray, image_size, pixels_per_cell))
        if i == 0:
            print(f"[hog] feature dim = {feats[0].shape[0]}  (sanity-check this before running the full set)")
        if (i + 1) % 500 == 0:
            print(f"  ...featurized {i + 1}/{len(dataset)}")
    return np.stack(feats, axis=0)


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------
#
# image_size and pixels_per_cell are now CLI-tunable (see main()), and both
# change the resulting feature dimension -- so they MUST be part of the cache
# key. Otherwise, changing --pixels_per_cell between runs could silently
# return a cached feature matrix from a different, incompatible dimension.

def _cache_filename(csv_path: str, num_classes, image_size, pixels_per_cell,
                     degradation_type, severity) -> str:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    parts = [base]
    if num_classes is not None:
        parts.append(f"n{num_classes}")
    parts.append(f"sz{image_size}_cell{pixels_per_cell}")
    if degradation_type is not None:
        parts.append(f"{degradation_type}_s{severity}")
    return "_".join(parts) + "_hog.npz"


def get_or_extract_hog_features(dataset: INatDataset, csv_path: str, cache_dir,
                                 image_size: int, pixels_per_cell: int,
                                 num_classes=None, degrade_fn=None,
                                 degradation_type=None, severity=None, seed=9517):
    """
    Loads cached HOG features from cache_dir if a matching, valid cache file
    exists; otherwise extracts features and writes the cache. cache_dir=None
    disables caching entirely (original slow-path behaviour).

    The image_ids stored in the cache are checked against the current
    dataset's image_ids before trusting the cache, so a stale or mismatched
    cache file is never silently used. image_size/pixels_per_cell are baked
    into the cache filename itself (see _cache_filename), so a dimension
    change never collides with an old cache file of a different shape.
    """
    if cache_dir is None:
        feats = dataset_to_hog_features(dataset, image_size, pixels_per_cell,
                                         degrade_fn, degradation_type, severity, seed)
        return feats, dataset.labels()

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, _cache_filename(csv_path, num_classes, image_size, pixels_per_cell,
                                    degradation_type, severity)
    )

    if os.path.exists(cache_path):
        print(f"[cache] found {cache_path}, checking image_id alignment...")
        cached = np.load(cache_path)
        if np.array_equal(cached["image_ids"], dataset.image_ids()):
            print(f"[cache] hit -- reusing cached HOG features ({cached['features'].shape[0]} images), skipping extraction")
            return cached["features"], cached["labels"]
        print("[cache] image_ids do not match the current dataset -- cache is stale, re-extracting")

    feats = dataset_to_hog_features(dataset, image_size, pixels_per_cell,
                                     degrade_fn, degradation_type, severity, seed)
    np.savez(cache_path, features=feats, image_ids=dataset.image_ids(), labels=dataset.labels())
    print(f"[cache] saved HOG features to {cache_path}")
    return feats, dataset.labels()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--data_root", default=".")
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--image_size", type=int, default=DEFAULT_HOG_IMAGE_SIZE,
                    help="Images are resized to (image_size, image_size) before HOG.")
    p.add_argument("--pixels_per_cell", type=int, default=DEFAULT_HOG_PIXELS_PER_CELL,
                    help="HOG cell size in pixels. Smaller (e.g. 8) captures finer detail "
                         "at the cost of a larger feature vector and slower extraction/training.")
    p.add_argument("--seed", type=int, default=9517)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None,
                    help="Directory to cache extracted HOG features (.npz). "
                         "If set, re-running with the same csv/num_classes/degradation "
                         "skips feature extraction entirely -- only useful when sweeping "
                         "RandomForest hyperparameters. Omit to disable caching.")

    p.add_argument("--degradation", default=None,
                    choices=[None, "gaussian_noise", "blur", "brightness", "jpeg_compression"])
    p.add_argument("--severity", type=int, default=None)
    p.add_argument("--load_model", default=None)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = INatDataset(args.train_csv, data_root=args.data_root,
                            num_classes=args.num_classes, seed=args.seed)
    test_ds = INatDataset(args.test_csv, data_root=args.data_root,
                           num_classes=args.num_classes, seed=args.seed)
    print(f"[data] train={len(train_ds)}  test={len(test_ds)}")

    degrade_fn = None
    if args.degradation is not None:
        assert args.severity is not None
        degrade_fn = load_degradation_fn()

    training_time = 0.0

    if args.load_model and os.path.exists(args.load_model):
        print(f"[model] loading cached RF from {args.load_model}")
        with open(args.load_model, "rb") as f:
            rf = pickle.load(f)
    else:
        with Timer() as t:
            print("[step] extracting HOG features for training set...")
            X_train, y_train = get_or_extract_hog_features(
                train_ds, args.train_csv, args.cache_dir,
                image_size=args.image_size, pixels_per_cell=args.pixels_per_cell,
                num_classes=args.num_classes, seed=args.seed,
            )

            print("[step] training RandomForestClassifier...")
            rf = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                n_jobs=args.n_jobs,
                random_state=args.seed,
            )
            rf.fit(X_train, y_train)
        training_time = t.elapsed
        with open(os.path.join(args.output_dir, "rf_model.pkl"), "wb") as f:
            pickle.dump(rf, f)

    with Timer() as t:
        print("[step] extracting HOG features for test set...")
        X_test, _ = get_or_extract_hog_features(
            test_ds, args.test_csv, args.cache_dir,
            image_size=args.image_size, pixels_per_cell=args.pixels_per_cell,
            num_classes=args.num_classes,
            degrade_fn=degrade_fn, degradation_type=args.degradation, severity=args.severity,
            seed=args.seed,
        )
        print("[step] running RF inference...")
        proba = rf.predict_proba(X_test)  # [N, num_classes_in_model]
        pred_cols = np.argmax(proba, axis=1)
        class_indices = rf.classes_  # again, column index != class_idx in general
        pred_labels = class_indices[pred_cols]
        top1_scores = proba[np.arange(len(proba)), pred_cols]
    inference_time = t.elapsed

    save_predictions_csv(
        args.output_dir,
        image_ids=test_ds.image_ids(),
        true_labels=test_ds.labels(),
        pred_labels=pred_labels,
        top1_scores=top1_scores,
        method_name=METHOD_NAME,
        split=test_ds.split_name(),  # reflects the actual csv (val/test), not hardcoded
    )
    save_scores_npz(
        args.output_dir,
        image_ids=test_ds.image_ids(),
        scores=proba,
        class_indices=class_indices,
    )
    save_runtime_json(
        args.output_dir,
        training_time_seconds=training_time,
        inference_time_seconds=inference_time,
        num_test_images=len(test_ds),
        extra_software={"scikit_image": __import__("skimage").__version__},
    )
    print("[done]")


if __name__ == "__main__":
    main()
