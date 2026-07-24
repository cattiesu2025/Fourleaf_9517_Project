"""
sift_bovw_svm.py
-----------------
Method 1: SIFT + Bag-of-Visual-Words + SVM

Feature caching design
-----------------------
The slow part of this pipeline is running cv2.SIFT_create().detectAndCompute()
on every image (image I/O + resize + keypoint detection + descriptor
computation). Clustering (MiniBatchKMeans) and building the final BoVW
histograms from already-extracted descriptors are comparatively fast.

So caching is placed at the *raw per-image descriptor* level, not at the
final histogram level. This means that once descriptors are cached for a
given (csv, num_classes, degradation) combination, you can freely sweep
--vocab_size (the hyperparameter you're most likely to tune) without ever
touching cv2.SIFT_create() again -- only clustering and histogram-building
re-run, which are fast.

Example usage:

  # Quick test on a small subset first (spec section 13.1 recommends this
  # before running on the full class set)
  python -m src.traditional.sift_bovw_svm \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/val.csv \
      --num_classes 50 \
      --vocab_size 200 \
      --output_dir outputs/traditional/sift_bovw_svm_smoketest

  # Full 500-class run (use val.csv during development; test.csv only at the end)
  python -m src.traditional.sift_bovw_svm \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/test.csv \
      --vocab_size 300 \
      --cache_dir outputs/traditional/sift_bovw_svm/feature_cache \
      --output_dir outputs/traditional/sift_bovw_svm

  # vocab_size sweep with descriptor caching: the first run extracts and
  # caches raw SIFT descriptors (slow, the actual bottleneck of this
  # pipeline); every subsequent run with a different --vocab_size reuses the
  # cached descriptors and only re-runs clustering + histogram + SVM (fast).
  python -m src.traditional.sift_bovw_svm \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/val.csv \
      --cache_dir outputs/traditional/sift_bovw_svm/feature_cache \
      --vocab_size 200 \
      --output_dir outputs/traditional/sift_bovw_svm_vocab200

  python -m src.traditional.sift_bovw_svm \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/val.csv \
      --cache_dir outputs/traditional/sift_bovw_svm/feature_cache \
      --vocab_size 400 \
      --output_dir outputs/traditional/sift_bovw_svm_vocab400

  # Robustness run (single degradation + severity combination)
  python -m src.traditional.sift_bovw_svm \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/test.csv \
      --vocab_size 300 \
      --cache_dir outputs/traditional/sift_bovw_svm/feature_cache \
      --load_vocab outputs/traditional/sift_bovw_svm/vocab.pkl \
      --load_model outputs/traditional/sift_bovw_svm/svm_model.pkl \
      --degradation gaussian_noise --severity 3 \
      --output_dir outputs/robustness/sift_bovw_svm/gaussian_noise/severity_3
"""

import os
import argparse
import pickle

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

from src.traditional.common_io import (
    INatDataset, Timer,
    save_predictions_csv, save_scores_npz, save_runtime_json,
    load_degradation_fn, degrade_pil_image,
)

METHOD_NAME = "sift_bovw_svm"


# ---------------------------------------------------------------------------
# Per-image descriptor extraction (the slow step)
# ---------------------------------------------------------------------------

def resize_for_sift(gray_img: np.ndarray, max_side: int = 256) -> np.ndarray:
    """SIFT does not need a full 224x224 image; capping the longest side speeds up extraction."""
    h, w = gray_img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        gray_img = cv2.resize(gray_img, (int(w * scale), int(h * scale)))
    return gray_img


def extract_sift_descriptors(gray_img: np.ndarray, max_desc_per_image: int = 100,
                              rng: np.random.RandomState = None) -> np.ndarray:
    """
    Extract SIFT descriptors from a single grayscale image, randomly capped
    at max_desc_per_image descriptors, to avoid an explosion in total
    descriptor count when clustering (see spec section 13.2).
    """
    sift = cv2.SIFT_create()
    _, desc = sift.detectAndCompute(gray_img, None)
    if desc is None:
        return np.zeros((0, 128), dtype=np.float32)
    if desc.shape[0] > max_desc_per_image:
        rng = rng or np.random.RandomState(0)
        idx = rng.choice(desc.shape[0], size=max_desc_per_image, replace=False)
        desc = desc[idx]
    return desc.astype(np.float32)


def dataset_to_sift_descriptors(dataset: INatDataset, max_desc_per_image: int,
                                 degrade_fn=None, degradation_type=None, severity=None, seed=9517):
    """
    Runs SIFT on every image in the dataset. Returns a Python list of
    per-image descriptor arrays (variable length, each [k_i, 128]), aligned
    1:1 with dataset order. This is the expensive step that caching targets.
    """
    rng = np.random.RandomState(seed)
    per_image_desc = []
    for i, rec in enumerate(dataset):
        img = dataset.load_image(rec, as_gray=False)  # raw RGB
        if degrade_fn is not None:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            pil_img = degrade_pil_image(degrade_fn, pil_img, degradation_type, severity, seed=seed)
            img = np.array(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = resize_for_sift(gray)
        desc = extract_sift_descriptors(gray, max_desc_per_image, rng)
        per_image_desc.append(desc)
        if (i + 1) % 500 == 0:
            print(f"  ...extracted descriptors for {i + 1}/{len(dataset)} images")
    return per_image_desc


# ---------------------------------------------------------------------------
# Descriptor caching
# ---------------------------------------------------------------------------
#
# Cached at the raw-descriptor level (variable-length per image), stored as
# an object-dtype numpy array inside an .npz (allow_pickle=True). This is
# independent of --vocab_size, so vocab_size sweeps hit the cache; only
# --max_desc_per_image, --degradation/--severity, --num_classes, and the csv
# file identity affect what gets cached.

def _cache_filename(csv_path: str, num_classes, max_desc_per_image, degradation_type, severity) -> str:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    parts = [base]
    if num_classes is not None:
        parts.append(f"n{num_classes}")
    parts.append(f"maxd{max_desc_per_image}")
    if degradation_type is not None:
        parts.append(f"{degradation_type}_s{severity}")
    return "_".join(parts) + "_sift.npz"


def get_or_extract_sift_descriptors(dataset: INatDataset, csv_path: str, cache_dir,
                                     max_desc_per_image: int, num_classes=None,
                                     degrade_fn=None, degradation_type=None, severity=None, seed=9517):
    """
    Loads cached per-image SIFT descriptors from cache_dir if a matching,
    valid cache file exists; otherwise extracts and writes the cache.
    cache_dir=None disables caching entirely (original slow-path behaviour).

    Returns a Python list of per-image descriptor arrays, aligned with
    dataset order. image_ids stored in the cache are checked against the
    current dataset before the cache is trusted.
    """
    if cache_dir is None:
        return dataset_to_sift_descriptors(
            dataset, max_desc_per_image, degrade_fn, degradation_type, severity, seed
        )

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, _cache_filename(csv_path, num_classes, max_desc_per_image, degradation_type, severity)
    )

    if os.path.exists(cache_path):
        print(f"[cache] found {cache_path}, checking image_id alignment...")
        cached = np.load(cache_path, allow_pickle=True)
        if np.array_equal(cached["image_ids"], dataset.image_ids()):
            print(f"[cache] hit -- reusing cached SIFT descriptors for {len(cached['descriptors'])} images, "
                  f"skipping SIFT extraction")
            return list(cached["descriptors"])
        print("[cache] image_ids do not match the current dataset -- cache is stale, re-extracting")

    per_image_desc = dataset_to_sift_descriptors(
        dataset, max_desc_per_image, degrade_fn, degradation_type, severity, seed
    )
    np.savez(
        cache_path,
        descriptors=np.array(per_image_desc, dtype=object),
        image_ids=dataset.image_ids(),
    )
    print(f"[cache] saved SIFT descriptors to {cache_path}")
    return per_image_desc


# ---------------------------------------------------------------------------
# Vocabulary + BoVW histograms (fast once descriptors exist)
# ---------------------------------------------------------------------------

def build_vocabulary(all_descriptors: np.ndarray, vocab_size: int,
                      max_total_desc: int, seed: int = 9517) -> MiniBatchKMeans:
    """
    Cluster the sampled descriptors with MiniBatchKMeans (not the standard
    KMeans) to build the visual vocabulary, as recommended in section 13.2.
    """
    rng = np.random.RandomState(seed)
    if all_descriptors.shape[0] > max_total_desc:
        idx = rng.choice(all_descriptors.shape[0], size=max_total_desc, replace=False)
        all_descriptors = all_descriptors[idx]

    print(f"[vocab] clustering {all_descriptors.shape[0]} descriptors into {vocab_size} words...")
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=seed, batch_size=1000, n_init=3)
    kmeans.fit(all_descriptors)
    return kmeans


def histogram_from_descriptors(desc: np.ndarray, kmeans: MiniBatchKMeans,
                                vocab_size: int) -> np.ndarray:
    """Already-extracted per-image descriptors -> L2-normalised BoVW histogram."""
    hist = np.zeros(vocab_size, dtype=np.float32)
    if desc is not None and len(desc) > 0:
        words = kmeans.predict(desc.astype(np.float32))
        for w in words:
            hist[w] += 1
    # L2 normalisation avoids scale differences caused by varying descriptor counts.
    hist = normalize(hist.reshape(1, -1), norm="l2")[0]
    return hist


def descriptors_to_bovw_features(per_image_desc, kmeans: MiniBatchKMeans, vocab_size: int) -> np.ndarray:
    """List of per-image descriptor arrays -> [N, vocab_size] feature matrix."""
    feats = [histogram_from_descriptors(d, kmeans, vocab_size) for d in per_image_desc]
    return np.stack(feats, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv", required=True, help="Use val.csv during development, test.csv only for the final run")
    p.add_argument("--data_root", default=".")
    p.add_argument("--num_classes", type=int, default=None, help="For quick testing, e.g. 50")
    p.add_argument("--vocab_size", type=int, default=300)
    p.add_argument("--max_desc_per_image", type=int, default=100)
    p.add_argument("--max_total_desc", type=int, default=500_000)
    p.add_argument("--svm_c", type=float, default=1.0,
                    help="LinearSVC regularization strength. Try 0.1 / 1 / 10 -- "
                         "lower C = more regularization, may help with only 40 "
                         "training images per class.")
    p.add_argument("--seed", type=int, default=9517)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None,
                    help="Directory to cache raw per-image SIFT descriptors (.npz). "
                         "If set, re-running with the same csv/num_classes/max_desc_per_image/"
                         "degradation skips SIFT extraction entirely and only re-runs "
                         "clustering + histogram + SVM -- useful when sweeping --vocab_size. "
                         "Omit to disable caching.")

    # robustness-related (optional)
    p.add_argument("--degradation", default=None,
                    choices=[None, "gaussian_noise", "blur", "brightness", "jpeg_compression"])
    p.add_argument("--severity", type=int, default=None)

    # reuse an already-trained vocab/model for inference (e.g. for robustness tests)
    p.add_argument("--load_vocab", default=None)
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
        assert args.severity is not None, "--severity must be given together with --degradation"
        degrade_fn = load_degradation_fn()

    training_time = 0.0

    # ---------------- Vocabulary ----------------
    if args.load_vocab and os.path.exists(args.load_vocab):
        print(f"[vocab] loading cached vocabulary from {args.load_vocab}")
        with open(args.load_vocab, "rb") as f:
            kmeans = pickle.load(f)
        train_desc = None  # not needed unless we also need to (re)train the SVM below
    else:
        with Timer() as t:
            print("[step] obtaining SIFT descriptors for training set...")
            train_desc = get_or_extract_sift_descriptors(
                train_ds, args.train_csv, args.cache_dir,
                max_desc_per_image=args.max_desc_per_image,
                num_classes=args.num_classes, seed=args.seed,
            )
            all_desc = np.concatenate([d for d in train_desc if d.shape[0] > 0], axis=0)
            kmeans = build_vocabulary(all_desc, args.vocab_size, args.max_total_desc, args.seed)
        training_time += t.elapsed
        with open(os.path.join(args.output_dir, "vocab.pkl"), "wb") as f:
            pickle.dump(kmeans, f)

    # ---------------- Training set BoVW features + SVM ----------------
    if args.load_model and os.path.exists(args.load_model):
        print(f"[model] loading cached SVM from {args.load_model}")
        with open(args.load_model, "rb") as f:
            svm = pickle.load(f)
    else:
        with Timer() as t:
            if train_desc is None:
                # vocab was loaded from disk but SVM was not -- still need
                # training descriptors to build histograms.
                print("[step] obtaining SIFT descriptors for training set...")
                train_desc = get_or_extract_sift_descriptors(
                    train_ds, args.train_csv, args.cache_dir,
                    max_desc_per_image=args.max_desc_per_image,
                    num_classes=args.num_classes, seed=args.seed,
                )
            print("[step] building BoVW histograms for training set...")
            X_train = descriptors_to_bovw_features(train_desc, kmeans, args.vocab_size)
            y_train = train_ds.labels()

            print(f"[step] training LinearSVC (C={args.svm_c})...")
            svm = LinearSVC(C=args.svm_c, max_iter=5000)
            svm.fit(X_train, y_train)
        training_time += t.elapsed
        with open(os.path.join(args.output_dir, "svm_model.pkl"), "wb") as f:
            pickle.dump(svm, f)

    # ---------------- Test set inference ----------------
    with Timer() as t:
        print("[step] obtaining SIFT descriptors for test set...")
        test_desc = get_or_extract_sift_descriptors(
            test_ds, args.test_csv, args.cache_dir,
            max_desc_per_image=args.max_desc_per_image,
            num_classes=args.num_classes,
            degrade_fn=degrade_fn, degradation_type=args.degradation, severity=args.severity,
            seed=args.seed,
        )
        print("[step] building BoVW histograms for test set...")
        X_test = descriptors_to_bovw_features(test_desc, kmeans, args.vocab_size)

        print("[step] running SVM inference...")
        scores = svm.decision_function(X_test)  # [N, num_classes_in_model] (not a probability)
        if scores.ndim == 1:
            # binary-classification edge case (should not happen with 500 classes, handled for safety)
            scores = np.stack([-scores, scores], axis=1)
        pred_cols = np.argmax(scores, axis=1)
        class_indices = svm.classes_  # important: column index != class_idx in general
        pred_labels = class_indices[pred_cols]
        top1_scores = scores[np.arange(len(scores)), pred_cols]
    inference_time = t.elapsed

    # ---------------- Save outputs ----------------
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
        scores=scores,
        class_indices=class_indices,
    )
    save_runtime_json(
        args.output_dir,
        training_time_seconds=training_time,
        inference_time_seconds=inference_time,
        num_test_images=len(test_ds),
        extra_software={"opencv": cv2.__version__},
    )
    print("[done]")


if __name__ == "__main__":
    main()
