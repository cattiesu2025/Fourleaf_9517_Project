from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.evaluation.contracts import EvaluationData


@dataclass(frozen=True)
class MetricResults:
    summary: dict[str, float]
    confusion: np.ndarray
    per_class: pd.DataFrame


def top_k_accuracy(data: EvaluationData, k: int = 5) -> float:
    if k < 1 or k > data.scores.shape[1]:
        raise ValueError(f"k must be between 1 and {data.scores.shape[1]}; got {k}")
    top_columns = np.argpartition(data.scores, -k, axis=1)[:, -k:]
    top_labels = data.class_indices[top_columns]
    correct = np.any(top_labels == data.y_true[:, None], axis=1)
    return float(np.mean(correct))


def calculate_metrics(data: EvaluationData) -> MetricResults:
    labels = data.labels
    precision, recall, f1, support = precision_recall_fscore_support(
        data.y_true,
        data.y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        data.y_true,
        data.y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    top1 = float(accuracy_score(data.y_true, data.y_pred))
    summary = {
        "top1_accuracy": top1,
        "top5_accuracy": top_k_accuracy(data, k=min(5, len(labels))),
        "overall_accuracy": top1,
        "balanced_accuracy": float(balanced_accuracy_score(data.y_true, data.y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }
    per_class = pd.DataFrame(
        {
            "class_idx": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(np.int64),
            "per_class_accuracy": recall,
        }
    )
    matrix = confusion_matrix(data.y_true, data.y_pred, labels=labels)
    return MetricResults(summary=summary, confusion=matrix, per_class=per_class)
