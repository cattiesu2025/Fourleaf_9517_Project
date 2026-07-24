"""Run transfer-learning ResNet18 inference and write E-compatible outputs.

Rewritten to match C's ACTUAL get_dataloader()/checkpoint/runtime.json
conventions (from src/scratch/predict.py). Robustness mode (--degradation
/--severity) is now wired straight through to get_dataloader, since C's
code confirms A's real dataloader.py handles degradation internally --
no more NotImplementedError placeholder.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import get_dataloader
from src.transfer.model import build_model, STRATEGY_TO_METHOD_NAME
from src.scratch.train import hardware_info, select_device, software_info, unpack_batch

PREDICTION_FIELDS = [
    "image_id",
    "true_label",
    "pred_label",
    "top1_score",
    "method_name",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with transfer-learning ResNet18.")
    parser.add_argument("--strategy", choices=["frozen", "finetuned", "layer4"], required=True)
    parser.add_argument("--use_attention", action="store_true", default=False)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                         help="Must match the value used in train.py for this checkpoint, "
                              "since it changes the fc layer's structure (fc.weight vs fc.1.weight).")
    parser.add_argument("--checkpoint", default=None)  # default derived from method_name below
    parser.add_argument("--test_csv", default="data/metadata/test.csv")
    parser.add_argument("--output_dir", default=None)  # default derived from method_name below
    parser.add_argument("--num_classes", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--degradation", default=None,
                         help="e.g. gaussian_noise / blur / brightness / jpeg_compression (E's degradation.py)")
    parser.add_argument("--severity", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


def load_checkpoint(path, device):
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    return checkpoint


def load_model(checkpoint_path, num_classes_arg, device, use_attention=False, num_heads=8, dropout_rate=0.0):
    checkpoint = load_checkpoint(checkpoint_path, device)
    num_classes = int(checkpoint.get("num_classes", num_classes_arg))
    model = build_model(num_classes, pretrained=False,
                         use_attention=use_attention, num_heads=num_heads,
                         dropout_rate=dropout_rate).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_runtime(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    if args.strategy not in STRATEGY_TO_METHOD_NAME:
        raise ValueError(f"'{args.strategy}' is not an official method name.")
    method_name = STRATEGY_TO_METHOD_NAME[args.strategy]
    if args.use_attention:
        method_name = f"{method_name}_attention_extra"

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/transfer/{method_name}")
    checkpoint_path = args.checkpoint or str(output_dir / "checkpoint_best.pth")

    # Robustness runs go under outputs/robustness/<method_name>/<degradation>/severity_<N>/
    # per Section 8 -- don't overwrite the clean-test outputs above.
    if args.degradation is not None:
        if args.severity is None:
            raise ValueError("--severity is required when --degradation is set")
        output_dir = Path(f"outputs/robustness/{method_name}/{args.degradation}/severity_{args.severity}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model = load_model(checkpoint_path, args.num_classes, device,
                        use_attention=args.use_attention, num_heads=args.num_heads,
                        dropout_rate=args.dropout_rate)

    loader = get_dataloader(
        split="test",
        csv_file=args.test_csv,
        transform_type="none",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        image_size=args.image_size,
        seed=args.seed,
        degradation_type=args.degradation,
        severity=args.severity,
    )

    prediction_rows: list[dict[str, Any]] = []
    all_image_ids: list[Any] = []
    all_scores: list[np.ndarray] = []
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="predict"), start=1):
            if args.max_batches is not None and batch_idx > args.max_batches:
                break

            images, labels, image_ids = unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            top1_scores, pred_labels = probs.max(dim=1)

            labels_np = labels.cpu().numpy()
            pred_np = pred_labels.cpu().numpy()
            top1_np = top1_scores.cpu().numpy()
            scores_np = probs.cpu().numpy().astype(np.float32)
            image_ids_list = image_ids.tolist() if hasattr(image_ids, "tolist") else list(image_ids)

            for image_id, true_label, pred_label, top1_score in zip(
                image_ids_list, labels_np, pred_np, top1_np
            ):
                prediction_rows.append({
                    "image_id": image_id,
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "top1_score": float(top1_score),
                    "method_name": method_name,
                    "split": "test",
                })
            all_image_ids.extend(image_ids_list)
            all_scores.append(scores_np)

    if not prediction_rows:
        raise RuntimeError("No predictions were produced")

    inference_time_seconds = time.perf_counter() - start_time
    scores = np.concatenate(all_scores, axis=0)
    class_indices = np.arange(scores.shape[1], dtype=np.int64)

    write_predictions(output_dir / "predictions.csv", prediction_rows)
    np.savez(
        output_dir / "scores.npz",
        image_ids=np.asarray(all_image_ids),
        scores=scores,
        class_indices=class_indices,
    )

    # Merge inference stats into runtime.json -- if this is a robustness run, there's no
    # training runtime.json alongside it (different directory), so start fresh in that case.
    runtime_path = output_dir / "runtime.json"
    runtime = read_runtime(runtime_path)
    runtime.update({
        "method_name": method_name,
        "inference_time_seconds": inference_time_seconds,
        "num_test_images": len(prediction_rows),
        "hardware": runtime.get("hardware", hardware_info(device)),
        "software": runtime.get("software", software_info()),
        "prediction_config": vars(args),
        "scores_type": "softmax_probability",
        "degradation": args.degradation,
        "severity": args.severity,
        "python": platform.python_version(),
    })
    with runtime_path.open("w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)

    print(f"Wrote predictions.csv, scores.npz, runtime.json to {output_dir}")
    print(f"Quick check test accuracy (E's evaluate.py is the source of truth): "
          f"{(np.array([r['pred_label'] for r in prediction_rows]) == np.array([r['true_label'] for r in prediction_rows])).mean():.4f}")


if __name__ == "__main__":
    main()
