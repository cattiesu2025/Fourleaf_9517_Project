"""Run transfer-learning ResNet18 inference and write standard artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import ArgumentParser
from src.common.prediction import run_torch_prediction, write_prediction_artifacts
from src.common.runtime import hardware_info, select_device, software_info
from src.data.dataloader import get_dataloader
from src.transfer.model import STRATEGY_TO_METHOD_NAME, build_model


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(description="Predict with transfer-learning ResNet18.")
    parser.add_argument("--strategy", choices=["frozen", "finetuned", "layer4"], required=True)
    parser.add_argument("--use_attention", action="store_true", default=False)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="Must match the value used in train.py for this checkpoint, "
        "since it changes the fc layer's structure (fc.weight vs fc.1.weight).",
    )
    parser.add_argument("--checkpoint", default=None)  # default derived from method_name below
    parser.add_argument("--test_csv", default="data/metadata/test.csv")
    parser.add_argument("--output_dir", default=None)  # default derived from method_name below
    parser.add_argument("--num_classes", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument(
        "--degradation",
        default=None,
        choices=["gaussian_noise", "blur", "brightness", "jpeg_compression"],
    )
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


def load_model(
    checkpoint_path, num_classes_arg, device, use_attention=False, num_heads=8, dropout_rate=0.0
):
    checkpoint = load_checkpoint(checkpoint_path, device)
    num_classes = int(checkpoint.get("num_classes", num_classes_arg))
    model = build_model(
        num_classes,
        pretrained=False,
        use_attention=use_attention,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if (args.degradation is None) != (args.severity is None):
        raise ValueError("--degradation and --severity must be provided together")

    if args.strategy not in STRATEGY_TO_METHOD_NAME:
        raise ValueError(f"'{args.strategy}' does not have a primary method name.")
    method_name = STRATEGY_TO_METHOD_NAME[args.strategy]
    if args.use_attention:
        method_name = f"{method_name}_attention_extra"

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(f"outputs/transfer/{method_name}")
    )
    checkpoint_path = args.checkpoint or str(output_dir / "checkpoint_best.pth")

    # Robustness runs go under outputs/robustness/<method_name>/<degradation>/severity_<N>/
    # per Section 8 -- don't overwrite the clean-test outputs above.
    if args.degradation is not None:
        if args.severity is None:
            raise ValueError("--severity is required when --degradation is set")
        output_dir = Path(
            f"outputs/robustness/{method_name}/{args.degradation}/severity_{args.severity}"
        )

    device = select_device(args.device)
    model = load_model(
        checkpoint_path,
        args.num_classes,
        device,
        use_attention=args.use_attention,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
    )

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
        method_name=method_name,
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

    print(f"Wrote predictions.csv, scores.npz, runtime.json to {output_dir}")
    print(f"Quick-check accuracy: {result.accuracy:.4f}")


if __name__ == "__main__":
    main()
