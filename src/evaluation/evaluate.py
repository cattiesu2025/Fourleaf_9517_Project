from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.confusion_matrix import (
    plot_confused_subset,
    save_confusion_matrix_outputs,
)
from src.evaluation.contracts import (
    load_class_names,
    load_evaluation_data,
    load_runtime,
)
from src.evaluation.error_analysis import failure_cases, top_confused_pairs
from src.evaluation.metrics import calculate_metrics


def evaluate_method(
    predictions_path: str | Path,
    scores_path: str | Path,
    output_dir: str | Path,
    *,
    classes_path: str | Path | None = None,
    runtime_path: str | Path | None = None,
    expected_num_classes: int | None = 500,
    expected_split: str = "test",
    strict_argmax: bool = True,
) -> dict[str, Any]:
    predictions_path = Path(predictions_path)
    scores_path = Path(scores_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_evaluation_data(
        predictions_path,
        scores_path,
        expected_num_classes=expected_num_classes,
        expected_split=expected_split,
        strict_argmax=strict_argmax,
    )
    class_names = load_class_names(classes_path, data.labels)
    runtime = load_runtime(runtime_path, len(data.predictions))
    results = calculate_metrics(data)

    per_class = results.per_class.copy()
    per_class.insert(
        1, "class_name", [class_names[int(label)] for label in per_class["class_idx"]]
    )
    confused = top_confused_pairs(results.confusion, data.labels, class_names)
    failures = failure_cases(data, class_names)

    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    confused.to_csv(output_dir / "top_confused_pairs.csv", index=False)
    failures.to_csv(output_dir / "failure_cases.csv", index=False)

    save_confusion_matrix_outputs(
        results.confusion,
        data.labels,
        class_names,
        output_dir,
        title=f"{data.method_name}: full confusion matrix",
    )
    plot_confused_subset(
        results.confusion,
        data.labels,
        class_names,
        confused,
        output_dir / "confusion_matrix_top_classes.png",
        title=f"{data.method_name}: most-confused class subset (row-normalised)",
    )

    payload: dict[str, Any] = {
        "schema_version": "1.1",
        "method_name": data.method_name,
        "split": data.split,
        "num_samples": int(len(data.predictions)),
        "num_classes": int(len(data.labels)),
        "metrics": results.summary,
        "runtime": runtime,
        "source_files": {
            "predictions": str(predictions_path),
            "scores": str(scores_path),
            "classes": str(classes_path) if classes_path is not None else None,
            "runtime": str(runtime_path) if runtime_path is not None else None,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and evaluate one COMP9517 method output."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--classes", type=Path, default=None)
    parser.add_argument("--runtime", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-num-classes", type=int, default=500)
    parser.add_argument("--split", default="test")
    parser.add_argument("--allow-argmax-mismatch", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    runtime_path = args.runtime
    if runtime_path is None:
        candidate = args.predictions.parent / "runtime.json"
        runtime_path = candidate if candidate.exists() else None
    payload = evaluate_method(
        args.predictions,
        args.scores,
        args.output,
        classes_path=args.classes,
        runtime_path=runtime_path,
        expected_num_classes=args.expected_num_classes,
        expected_split=args.split,
        strict_argmax=not args.allow_argmax_mismatch,
    )
    metrics = payload["metrics"]
    print(
        f"Evaluated {payload['method_name']} ({payload['num_samples']} samples): "
        f"Top-1={metrics['top1_accuracy']:.4f}, Top-5={metrics['top5_accuracy']:.4f}, "
        f"Macro-F1={metrics['macro_f1']:.4f}"
    )
    print(f"Outputs written to {args.output}")


if __name__ == "__main__":
    main()
