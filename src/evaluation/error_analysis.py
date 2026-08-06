from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.contracts import EvaluationData


def top_confused_pairs(
    matrix: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
    limit: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    supports = matrix.sum(axis=1)
    for true_position, true_label in enumerate(labels):
        for pred_position, pred_label in enumerate(labels):
            if true_position == pred_position:
                continue
            count = int(matrix[true_position, pred_position])
            if count == 0:
                continue
            rows.append(
                {
                    "true_label": int(true_label),
                    "true_class_name": class_names[int(true_label)],
                    "pred_label": int(pred_label),
                    "pred_class_name": class_names[int(pred_label)],
                    "count": count,
                    "fraction_of_true_class": float(count / supports[true_position])
                    if supports[true_position]
                    else 0.0,
                }
            )
    columns = [
        "true_label",
        "true_class_name",
        "pred_label",
        "pred_class_name",
        "count",
        "fraction_of_true_class",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["count", "fraction_of_true_class"], ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def failure_cases(data: EvaluationData, class_names: dict[int, str]) -> pd.DataFrame:
    failures = data.predictions.loc[data.y_true != data.y_pred].copy()
    extra_columns = [
        "true_class_name",
        "pred_class_name",
        "true_score",
        "pred_score",
        "score_margin",
    ]
    if failures.empty:
        return pd.DataFrame(columns=[*data.predictions.columns, *extra_columns])

    class_column = {int(label): position for position, label in enumerate(data.class_indices)}
    failure_positions = np.flatnonzero(data.y_true != data.y_pred)
    true_scores = np.asarray(
        [data.scores[row, class_column[int(data.y_true[row])]] for row in failure_positions],
        dtype=float,
    )
    pred_scores = np.asarray(
        [data.scores[row, class_column[int(data.y_pred[row])]] for row in failure_positions],
        dtype=float,
    )
    failures["true_class_name"] = [class_names[int(label)] for label in failures["true_label"]]
    failures["pred_class_name"] = [class_names[int(label)] for label in failures["pred_label"]]
    failures["true_score"] = true_scores
    failures["pred_score"] = pred_scores
    failures["score_margin"] = pred_scores - true_scores
    return failures.sort_values("score_margin", ascending=False).reset_index(drop=True)
