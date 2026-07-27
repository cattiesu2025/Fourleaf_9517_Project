"""Adapters that expose the final B/C/D models to the shared robustness runner."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.transforms import get_eval_transform
from src.scratch.model import build_resnet18_scratch
from src.traditional.hog_random_forest import extract_hog_feature
from src.traditional.sift_bovw_svm import (
    descriptors_to_bovw_features,
    extract_sift_descriptors,
    resize_for_sift,
)
from src.transfer.model import build_model as build_transfer_model


def _load_pickle(path: str | Path) -> Any:
    """Load a trusted team-produced pickle artifact.

    Pickle can execute code while loading.  This helper is intentionally only
    used for the model files produced by the project and supplied by member B.
    """

    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


class HOGRandomForestPredictor:
    method_name = "hog_random_forest"

    def __init__(
        self,
        model: Any,
        *,
        image_size: int = 128,
        pixels_per_cell: int = 16,
    ) -> None:
        self.model = model
        self.image_size = int(image_size)
        self.pixels_per_cell = int(pixels_per_cell)
        self.class_indices = np.asarray(model.classes_, dtype=np.int64)

    @classmethod
    def from_pickle(cls, path: str | Path, **kwargs: Any) -> "HOGRandomForestPredictor":
        return cls(_load_pickle(path), **kwargs)

    def predict_scores(self, image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        feature = extract_hog_feature(
            gray, self.image_size, self.pixels_per_cell
        ).reshape(1, -1)
        return np.asarray(self.model.predict_proba(feature)[0], dtype=np.float64)

    def predict_scores_batch(
        self, images: list[Image.Image], image_ids: list[str]
    ) -> np.ndarray:
        del image_ids
        features = []
        for image in images:
            rgb = np.asarray(image.convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            features.append(
                extract_hog_feature(gray, self.image_size, self.pixels_per_cell)
            )
        return np.asarray(self.model.predict_proba(np.stack(features)), dtype=np.float64)


class SIFTBoVWSVMPredictor:
    method_name = "sift_bovw_svm"

    def __init__(
        self,
        vocabulary: Any,
        model: Any,
        *,
        max_desc_per_image: int = 200,
        feature_seed: int = 9517,
    ) -> None:
        self.vocabulary = vocabulary
        self.model = model
        self.max_desc_per_image = int(max_desc_per_image)
        self.feature_seed = int(feature_seed)
        self.vocab_size = int(vocabulary.n_clusters)
        self.class_indices = np.asarray(model.classes_, dtype=np.int64)
        self._rng = np.random.RandomState(self.feature_seed)

    @classmethod
    def from_pickles(
        cls, vocabulary_path: str | Path, model_path: str | Path, **kwargs: Any
    ) -> "SIFTBoVWSVMPredictor":
        return cls(_load_pickle(vocabulary_path), _load_pickle(model_path), **kwargs)

    def begin_run(self, degradation_type: str, severity: int) -> None:
        """Reset descriptor sampling so every severity uses the clean-run order."""

        del degradation_type, severity
        self._rng = np.random.RandomState(self.feature_seed)

    def _feature(self, image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = resize_for_sift(gray)
        descriptors = extract_sift_descriptors(
            gray, self.max_desc_per_image, rng=self._rng
        )
        return descriptors_to_bovw_features(
            [descriptors], self.vocabulary, self.vocab_size
        )[0]

    def predict_scores(self, image: Image.Image) -> np.ndarray:
        feature = self._feature(image).reshape(1, -1)
        return np.asarray(self.model.decision_function(feature)[0], dtype=np.float64)

    def predict_scores_batch(
        self, images: list[Image.Image], image_ids: list[str]
    ) -> np.ndarray:
        features = np.stack(
            [
                self._feature(image)
                for image, _image_id in zip(images, image_ids, strict=True)
            ]
        )
        scores = np.asarray(self.model.decision_function(features), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return scores


class TorchResNetPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        method_name: str,
        num_classes: int,
        device: torch.device,
        image_size: int = 224,
    ) -> None:
        self.model = model.to(device).eval()
        self.method_name = method_name
        self.class_indices = np.arange(num_classes, dtype=np.int64)
        self.device = device
        self.transform = get_eval_transform(image_size=image_size)

    @staticmethod
    def _checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"unexpected checkpoint format: {checkpoint_path}")
        return checkpoint

    @classmethod
    def from_scratch_checkpoint(
        cls,
        path: str | Path,
        *,
        method_name: str = "resnet18_scratch_basic_aug_sgd",
        num_classes: int = 500,
        device: str = "auto",
        image_size: int = 224,
    ) -> "TorchResNetPredictor":
        resolved_device = resolve_device(device)
        checkpoint = cls._checkpoint(path, resolved_device)
        actual_classes = int(checkpoint.get("num_classes", num_classes))
        model = build_resnet18_scratch(num_classes=actual_classes)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        return cls(
            model,
            method_name=method_name,
            num_classes=actual_classes,
            device=resolved_device,
            image_size=image_size,
        )

    @classmethod
    def from_transfer_checkpoint(
        cls,
        path: str | Path,
        *,
        method_name: str = "resnet18_pretrained_finetuned",
        num_classes: int = 500,
        device: str = "auto",
        image_size: int = 224,
        use_attention: bool = False,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
    ) -> "TorchResNetPredictor":
        resolved_device = resolve_device(device)
        checkpoint = cls._checkpoint(path, resolved_device)
        actual_classes = int(checkpoint.get("num_classes", num_classes))
        model = build_transfer_model(
            actual_classes,
            pretrained=False,
            use_attention=use_attention,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        return cls(
            model,
            method_name=method_name,
            num_classes=actual_classes,
            device=resolved_device,
            image_size=image_size,
        )

    def predict_scores(self, image: Image.Image) -> np.ndarray:
        return self.predict_scores_batch([image], ["0"])[0]

    def predict_scores_batch(
        self, images: list[Image.Image], image_ids: list[str]
    ) -> np.ndarray:
        del image_ids
        tensor = torch.stack([self.transform(image) for image in images]).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)
        return probabilities.detach().cpu().numpy().astype(np.float64)
