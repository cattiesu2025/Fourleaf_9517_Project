"""Run scratch ResNet18 inference and write standard project artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import ArgumentParser
from src.common.prediction import run_torch_prediction, write_prediction_artifacts
from src.common.runtime import hardware_info, select_device, software_info
from src.data.dataloader import get_dataloader
from src.scratch.model import build_resnet18_scratch


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(description="Predict with scratch ResNet18.")
    parser.add_argument("--method_name", default="resnet18_scratch_basic_aug")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_csv", default="data/metadata/test.csv")
    parser.add_argument("--output_dir", default="outputs/scratch/resnet18_scratch_basic_aug")
    parser.add_argument("--num_classes", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--degradation", default=None)
    parser.add_argument("--severity", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    return checkpoint


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    checkpoint = load_checkpoint(args.checkpoint, device)
    num_classes = int(checkpoint.get("num_classes", args.num_classes))
    model = build_resnet18_scratch(num_classes=num_classes).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if (args.degradation is None) != (args.severity is None):
        raise ValueError("--degradation and --severity must be provided together")
    output_dir = Path(args.output_dir)

    device = select_device(args.device)
    model = load_model(args, device)
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

    result = run_torch_prediction(
        model,
        loader,
        device=device,
        method_name=args.method_name,
        max_batches=args.max_batches,
    )
    write_prediction_artifacts(
        output_dir,
        result,
        hardware=hardware_info(device),
        software=software_info(),
        prediction_config=vars(args),
        degradation=args.degradation,
        severity=args.severity,
    )
    print(f"Wrote prediction artifacts to {output_dir}")
    print(f"Quick-check accuracy: {result.accuracy:.4f}")


if __name__ == "__main__":
    main()
