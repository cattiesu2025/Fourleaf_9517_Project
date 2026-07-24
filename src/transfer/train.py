"""Train ResNet18 transfer-learning models (frozen / finetuned).

Rewritten to match C's ACTUAL get_dataloader() call signature and
runtime.json/checkpoint conventions (from C's src/scratch/train.py),
so D's outputs are format-identical to C's for E's aggregation.

Reuses hardware_info/software_info/select_device/set_seed from
src.scratch.train rather than reimplementing them -- if this dependency
feels backwards (transfer importing from scratch), suggest to the group
moving these into src/common/utils.py; not done here since that would
require editing C's file too.
"""

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
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import get_dataloader
from src.transfer.model import build_model, set_finetune_strategy, STRATEGY_TO_METHOD_NAME
# Reused from C so runtime.json / hardware info are byte-identical in format:
from src.scratch.train import hardware_info, select_device, software_info, set_seed, unpack_batch

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
    parser = argparse.ArgumentParser(description="Train transfer-learning ResNet18.")
    parser.add_argument("--strategy", choices=["frozen", "finetuned", "layer4"], required=True)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--use_attention", action="store_true", default=False,
                         help="Extra ablation: insert multi-head self-attention after layer4. "
                              "NOT an official method_name -- outputs go to a separate directory.")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--train_csv", default="data/metadata/train.csv")
    parser.add_argument("--val_csv", default="data/metadata/val.csv")
    parser.add_argument("--transform_type", choices=["none", "basic_aug", "strong_aug"], default="basic_aug")
    parser.add_argument("--num_classes", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)  # None -> strategy-dependent default below
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                         help="Dropout before fc layer, e.g. 0.4 -- regularization for overfitting.")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                         help="e.g. 0.1 -- softens hard one-hot targets, another overfitting knob.")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default="cosine")
    parser.add_argument("--sampler", choices=["none", "weighted_random"], default="none")
    parser.add_argument("--seed", type=int, default=9517)  # training-run seed (weight init/shuffle), separate from A's data-split seed (500)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output_dir", default=None)  # default derived from method_name below
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


def build_optimizer(args: argparse.Namespace, trainable_params) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    return AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    return CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None, use_amp=False):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total_seen = 0.0, 0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_epoch, best_val_accuracy, args, method_name):
    payload = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "model_state_dict": model.state_dict(),  # matches C's key name, not "model_state"
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "num_classes": args.num_classes,
        "method_name": method_name,
        "strategy": args.strategy,
        "training_config": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()

    if args.strategy not in STRATEGY_TO_METHOD_NAME:
        raise ValueError(
            f"'{args.strategy}' is not an official method name. Only 'frozen' and "
            f"'finetuned' are official (see 命名清单); 'layer4' is an extra ablation."
        )
    method_name = STRATEGY_TO_METHOD_NAME[args.strategy]
    if args.use_attention:
        # Extra ablation, not an official method_name -- keep it out of the
        # official outputs/transfer/<method_name>/ directories entirely.
        method_name = f"{method_name}_attention_extra"
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/transfer/{method_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
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

    model = build_model(args.num_classes, pretrained=args.pretrained,
                         use_attention=args.use_attention, num_heads=args.num_heads,
                         dropout_rate=args.dropout_rate).to(device)
    set_finetune_strategy(model, args.strategy)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    lr = args.lr if args.lr is not None else (1e-3 if args.strategy == "frozen" else 1e-4)
    args.lr = lr  # so it's recorded correctly in training_config
    optimizer = build_optimizer(args, trainable_params)
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device,
                                           optimizer=optimizer, max_batches=args.max_train_batches, use_amp=use_amp)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device,
                                       optimizer=None, max_batches=args.max_val_batches, use_amp=False)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        row = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr(optimizer),
            "epoch_time_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        write_history(output_dir / "training_history.csv", history)

        if val_acc > best_val_accuracy:
            best_val_accuracy, best_epoch = val_acc, epoch
            save_checkpoint(output_dir / "checkpoint_best.pth", model, optimizer, scheduler,
                             epoch, best_epoch, best_val_accuracy, args, method_name)
        save_checkpoint(output_dir / "checkpoint_last.pth", model, optimizer, scheduler,
                         epoch, best_epoch, best_val_accuracy, args, method_name)

        print(f"[{method_name}] epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    runtime = {
        "method_name": method_name,
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

    print(f"\nCheckpoints + training_history.csv + runtime.json written to {output_dir}")


if __name__ == "__main__":
    main()
