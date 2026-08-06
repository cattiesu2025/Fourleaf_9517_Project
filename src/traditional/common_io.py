"""
common_io.py
------------
Shared utilities for the two traditional-method pipelines
(SIFT+BoVW+SVM, HOG+RandomForest):

  1. Dataset loading (reads data/metadata/*.csv)
  2. Degradation call wrapper for real-time robustness evaluation
  3. Output saving (predictions.csv / scores.npz / runtime.json)

No ImageNet normalisation is done here. These pipelines only resize images
and perform the grayscale conversion needed by each feature extractor.
"""

import json
import os
import platform
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class INatRecord:
    image_id: int
    image_path: str
    class_idx: int


class INatDataset:
    """
    Minimal dataset wrapper, no torch dependency, pure PIL + numpy, so it's
    easy to plug into the sklearn pipelines.

    csv_file must be one of data/metadata/{train,val,test,longtail_train}.csv
    columns: image_id, image_path, original_class_id, class_idx, class_name, split

    data_root: image_path is relative to this directory, which normally points
               to the project root.
    """

    def __init__(
        self,
        csv_file: str,
        data_root: str = ".",
        num_classes: Optional[int] = None,
        seed: int = 9517,
    ):
        df = pd.read_csv(csv_file)

        # Optional small class subset for quick feature and memory checks.
        # Usage: INatDataset(..., num_classes=50) randomly picks a subset of
        # class_idx values. This does NOT re-split the data (still allowed
        # under the contract, since it's just a subset filter -- it doesn't
        # change the image_id/class_idx mapping or mix splits).
        if num_classes is not None:
            rng = np.random.RandomState(seed)
            all_classes = sorted(df["class_idx"].unique())
            if num_classes < len(all_classes):
                chosen = rng.choice(all_classes, size=num_classes, replace=False)
                df = df[df["class_idx"].isin(chosen)].reset_index(drop=True)

        self.df = df
        self.data_root = data_root
        self.records: List[INatRecord] = [
            INatRecord(int(r.image_id), r.image_path, int(r.class_idx))
            for r in df.itertuples(index=False)
        ]

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def load_image(self, record: INatRecord, as_gray: bool = False) -> np.ndarray:
        """Load the raw RGB image (PIL -> numpy), no model-specific preprocessing."""
        path = os.path.join(self.data_root, record.image_path)
        img = Image.open(path).convert("RGB")
        if as_gray:
            img = img.convert("L")
        return np.array(img)

    def image_ids(self) -> np.ndarray:
        return np.array([r.image_id for r in self.records], dtype=np.int64)

    def labels(self) -> np.ndarray:
        return np.array([r.class_idx for r in self.records], dtype=np.int64)

    def split_name(self) -> str:
        """
        Returns the actual split value found in the csv's 'split' column
        (e.g. 'train' / 'val' / 'test'), instead of letting callers hardcode
        it. Each metadata csv (train.csv / val.csv / test.csv / ...) is
        expected to contain exactly one split value throughout; if the csv
        was constructed incorrectly and mixes splits, this raises rather
        than silently picking one.
        """
        unique = self.df["split"].unique()
        if len(unique) != 1:
            raise ValueError(
                f"Expected exactly one split value in the csv, found: {list(unique)}. "
                f"Each metadata csv should contain a single split (train/val/test)."
            )
        return str(unique[0])


# ---------------------------------------------------------------------------
# Degradation call wrapper
# ---------------------------------------------------------------------------


def load_degradation_fn() -> Callable:
    """
    Import the canonical degradation implementation. Traditional models call
    it without changing its parameters or insertion point.
    """
    try:
        from src.evaluation.degradation import apply_degradation  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Could not find apply_degradation in src/evaluation/degradation.py. "
            "Restore the shared implementation before running robustness."
        ) from e
    return apply_degradation


def degrade_pil_image(
    apply_degradation_fn,
    pil_img: Image.Image,
    degradation_type: str,
    severity: int,
    seed: Optional[int] = None,
):
    """
    Degradation must happen "after loading the raw image, before each model's
    own preprocessing." The pil_img passed here must be the image straight
    off disk, before any resize/grayscale/normalisation.
    """
    return apply_degradation_fn(pil_img, degradation_type, severity, seed=seed)


# ---------------------------------------------------------------------------
# Output saving: predictions.csv / scores.npz / runtime.json
# ---------------------------------------------------------------------------


def save_predictions_csv(
    out_dir: str,
    image_ids: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    top1_scores: np.ndarray,
    method_name: str,
    split: str = "test",
):
    """
    Strictly follows section 5.1 format:
    image_id,true_label,pred_label,top1_score,method_name,split

    split should be the actual split the inference was run on (typically
    obtained via dataset.split_name()), NOT a hardcoded literal -- otherwise
    a predictions.csv produced from val.csv would incorrectly claim split=test.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_label": true_labels,
            "pred_label": pred_labels,
            "top1_score": top1_scores,
            "method_name": method_name,
            "split": split,
        }
    )
    out_path = os.path.join(out_dir, "predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"[saved] {out_path}  ({len(df)} rows)")
    return out_path


def save_scores_npz(
    out_dir: str,
    image_ids: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray,
):
    """
    Strictly follows section 5.2 format.

    scores: [N, num_classes_the_model_actually_outputs] (SVM/RF may not cover
            all 500 classes in classes_ if a class has zero samples in the
            training subset).
    class_indices: the true class_idx for column j, taken from model.classes_.
                   Never assume column j == class_idx j.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scores.npz")
    np.savez(
        out_path,
        image_ids=image_ids,
        scores=scores,
        class_indices=class_indices,
    )
    print(f"[saved] {out_path}  scores shape={scores.shape}")
    return out_path


def save_runtime_json(
    out_dir: str,
    training_time_seconds: float,
    inference_time_seconds: float,
    num_test_images: int,
    extra_software: Optional[dict] = None,
):
    """Strictly follows section 5.3 traditional-method format (CPU, no GPU field)."""
    os.makedirs(out_dir, exist_ok=True)
    software = {"python": platform.python_version()}
    if extra_software:
        software.update(extra_software)

    payload = {
        "training_time_seconds": round(training_time_seconds, 2),
        "inference_time_seconds": round(inference_time_seconds, 2),
        "num_test_images": num_test_images,
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor() or "unknown",
            "gpu": None,
            "ram_gb": None,  # optional: fill in with psutil.virtual_memory().total / 1e9
        },
        "software": software,
    }
    out_path = os.path.join(out_dir, "runtime.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {out_path}")
    return out_path


class Timer:
    """Small helper: with Timer() as t: ... ; t.elapsed"""

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
