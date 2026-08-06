"""Train ResNet18 from scratch for the augmentation ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import ArgumentParser
from src.common.runtime import (
    hardware_info,
    select_device,
    set_seed,
    software_info,
    unpack_batch,
)
from src.data.dataloader import get_dataloader
from src.scratch.model import build_resnet18_scratch

HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "lr",
    "epoch_time_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(description="Train scratch ResNet18.")
    parser.add_argument("--method_name", default="resnet18_scratch_basic_aug")
    parser.add_argument(
        "--transform_type",
        choices=["none", "basic_aug", "strong_aug"],
        default="basic_aug",
    )
    parser.add_argument("--train_csv", default="data/metadata/train.csv")
    parser.add_argument("--val_csv", default="data/metadata/val.csv")
    parser.add_argument("--output_dir", default="outputs/scratch/resnet18_scratch_basic_aug")
    parser.add_argument("--num_classes", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default="cosine")
    parser.add_argument("--sampler", choices=["none", "weighted_random"], default="none")
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> Any:
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    return CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    progress = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for batch_idx, batch in enumerate(progress, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        images, labels, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = int(labels.size(0))
        total_seen += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        progress.set_postfix(loss=total_loss / total_seen, acc=total_correct / total_seen)

    if total_seen == 0:
        raise RuntimeError("No samples were processed in this epoch")
    return total_loss / total_seen, total_correct / total_seen


def write_history(history_path: Path, rows: list[dict[str, Any]]) -> None:
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_epoch: int,
    best_val_accuracy: float,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "num_classes": args.num_classes,
        "method_name": args.method_name,
        "training_config": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    train_loader = get_dataloader(
        split="train",
        csv_file=args.train_csv,
        transform_type=args.transform_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=args.sampler,
        image_size=args.image_size,
        seed=args.seed,
    )
    val_loader = get_dataloader(
        split="val",
        csv_file=args.val_csv,
        transform_type="none",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        image_size=args.image_size,
        seed=args.seed,
    )

    model = build_resnet18_scratch(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            use_amp=use_amp,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
            use_amp=False,
        )

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr(optimizer),
            "epoch_time_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        write_history(output_dir / "training_history.csv", history)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            save_checkpoint(
                output_dir / "checkpoint_best.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_epoch,
                best_val_accuracy,
                args,
            )

        save_checkpoint(
            output_dir / "checkpoint_last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_epoch,
            best_val_accuracy,
            args,
        )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    runtime = {
        "method_name": args.method_name,
        "training_time_seconds": time.perf_counter() - start_time,
        "num_train_images": len(train_loader.dataset),
        "num_val_images": len(val_loader.dataset),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "checkpoint_selection": "best_val_accuracy",
        "hardware": hardware_info(device),
        "software": software_info(),
        "training_config": vars(args),
    }
    with (output_dir / "runtime.json").open("w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)


if __name__ == "__main__":
    main()
