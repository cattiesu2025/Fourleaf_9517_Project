"""
src/transfer/gradcam_crop_fusion.py
--------------------------------------
Two-stage inference pipeline: classify -> Grad-CAM localize -> crop the
activated region -> re-classify the crop -> fuse both predictions.

IMPORTANT (label leakage): Grad-CAM is generated using the MODEL'S OWN
predicted class (argmax), never the ground-truth label, at every step --
this must hold at test time or the "improvement" would be meaningless.

This does NOT require retraining -- it reuses the already-trained
finetuned checkpoint for both passes (stage 1 on the full image, stage
2 on the Grad-CAM-cropped region), fusing their softmax outputs. This
is an extra, non-official ablation (method_name gets a distinct suffix,
not one of the two official names).

Usage:
    python src/transfer/gradcam_crop_fusion.py --strategy finetuned

Output goes to outputs/transfer/resnet18_pretrained_finetuned_gradcam_crop_fusion_extra/
(predictions.csv, scores.npz, runtime.json -- same schema as other methods,
plus a metrics.json comparing fused vs stage-1-only accuracy).
"""

import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.transfer.model import build_model, STRATEGY_TO_METHOD_NAME
from src.transfer.gradcam import GradCAM
from src.data.dataloader import get_dataloader


def heatmap_to_bbox(heatmap: np.ndarray, percentile: float = 70.0,
                     padding_frac: float = 0.15, min_size_frac: float = 0.3):
    """
    Thresholds a (H, W) Grad-CAM heatmap (values in [0,1]) at the given
    percentile, finds the bounding box of the activated region, adds
    padding, and enforces a minimum crop size (as a fraction of the
    full image) so we never crop down to a sliver.

    Returns (y0, y1, x0, x1) in pixel coordinates, or None if the
    activated region is degenerate (covers almost the whole image --
    cropping wouldn't add information).
    """
    H, W = heatmap.shape
    thresh = np.percentile(heatmap, percentile)
    mask = heatmap > thresh  # strict > -- avoids swallowing large zero-plateaus sitting at the threshold
    # (>= would misfire whenever many pixels are exactly at the threshold value, which is
    # common for sparse/well-localized Grad-CAM heatmaps with large flat zero backgrounds)
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # Skip cropping if the activated region already covers almost the whole image
    if (y1 - y0) > 0.9 * H and (x1 - x0) > 0.9 * W:
        return None

    # Padding
    pad_y = int((y1 - y0) * padding_frac)
    pad_x = int((x1 - x0) * padding_frac)
    y0, y1 = max(0, y0 - pad_y), min(H, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(W, x1 + pad_x)

    # Enforce minimum crop size (centered) so tiny/degenerate boxes don't collapse the image
    min_h, min_w = int(H * min_size_frac), int(W * min_size_frac)
    if (y1 - y0) < min_h:
        cy = (y0 + y1) // 2
        y0, y1 = max(0, cy - min_h // 2), min(H, cy + min_h // 2)
    if (x1 - x0) < min_w:
        cx = (x0 + x1) // 2
        x0, x1 = max(0, cx - min_w // 2), min(W, cx + min_w // 2)

    return int(y0), int(y1), int(x0), int(x1)


def crop_and_resize(img_tensor: torch.Tensor, bbox, out_size: int) -> torch.Tensor:
    """img_tensor: (1, 3, H, W), already normalized. Returns (1, 3, out_size, out_size)."""
    y0, y1, x0, x1 = bbox
    cropped = img_tensor[:, :, y0:y1, x0:x1]
    return F.interpolate(cropped, size=(out_size, out_size), mode="bilinear", align_corners=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["frozen", "finetuned", "layer4"], required=True)
    parser.add_argument("--checkpoint_dir", default=None,
                         help="Directory containing checkpoint_best.pth to use for BOTH passes "
                              "(defaults to outputs/transfer/<official_method_name>).")
    parser.add_argument("--test_csv", default="data/metadata/test.csv")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--percentile", type=float, default=70.0,
                         help="Heatmap threshold percentile for the bounding box.")
    parser.add_argument("--padding_frac", type=float, default=0.15)
    parser.add_argument("--min_size_frac", type=float, default=0.3)
    parser.add_argument("--fusion_weight", type=float, default=0.5,
                         help="Weight on stage-2 (cropped) probs; stage-1 gets (1 - this).")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_images", type=int, default=None, help="For a quick smoke test.")
    args = parser.parse_args()

    official_method_name = STRATEGY_TO_METHOD_NAME[args.strategy]
    checkpoint_dir = args.checkpoint_dir or f"outputs/transfer/{official_method_name}"
    output_dir = args.output_dir or f"outputs/transfer/{official_method_name}_gradcam_crop_fusion_extra"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(os.path.join(checkpoint_dir, "checkpoint_best.pth"), map_location=device)
    model = build_model(ckpt["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    num_classes = ckpt["num_classes"]

    # batch_size=1: Grad-CAM's hook/backward logic assumes a single image per forward pass.
    loader = get_dataloader(
        split="test", csv_file=args.test_csv, transform_type="none",
        batch_size=1, shuffle=False, image_size=args.image_size, seed=args.seed,
    )

    image_ids, true_labels = [], []
    stage1_scores, fused_scores = [], []
    num_cropped = 0
    start_time = time.perf_counter()

    for i, (img, label, image_id) in enumerate(loader):
        if args.max_images is not None and i >= args.max_images:
            break
        img = img.to(device)

        # --- Stage 1: classify the full image ---
        with torch.no_grad():
            logits1 = model(img)
            probs1 = torch.softmax(logits1, dim=1)

        # --- Grad-CAM using the model's OWN predicted class (never the true label) ---
        cam = GradCAM(model, target_layer=model.layer4[-1])
        heatmap, pred_class = cam.generate(img, class_idx=None)
        cam.remove_hooks()

        bbox = heatmap_to_bbox(heatmap, args.percentile, args.padding_frac, args.min_size_frac)

        if bbox is not None:
            num_cropped += 1
            cropped_img = crop_and_resize(img, bbox, args.image_size)
            with torch.no_grad():
                logits2 = model(cropped_img)
                probs2 = torch.softmax(logits2, dim=1)
            fused = (1 - args.fusion_weight) * probs1 + args.fusion_weight * probs2
        else:
            fused = probs1  # degenerate box -- fall back to stage-1 only

        iid = image_id.item() if hasattr(image_id, "item") else image_id[0]
        image_ids.append(iid)
        true_labels.append(int(label.item()))
        stage1_scores.append(probs1[0].detach().cpu().numpy())
        fused_scores.append(fused[0].detach().cpu().numpy())

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1} images...")

    inference_time_seconds = time.perf_counter() - start_time

    stage1_scores = np.stack(stage1_scores)
    fused_scores = np.stack(fused_scores)
    true_labels = np.array(true_labels)
    class_indices = np.arange(num_classes)

    stage1_pred = stage1_scores.argmax(axis=1)
    fused_pred = fused_scores.argmax(axis=1)
    stage1_acc = float((stage1_pred == true_labels).mean())
    fused_acc = float((fused_pred == true_labels).mean())

    # --- Write predictions.csv / scores.npz (fused = the "official" output of this method) ---
    method_name = f"{official_method_name}_gradcam_crop_fusion_extra"
    with open(os.path.join(output_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "true_label", "pred_label", "top1_score", "method_name", "split"])
        for iid, tl, pl, row_scores in zip(image_ids, true_labels, fused_pred, fused_scores):
            writer.writerow([iid, tl, pl, float(row_scores.max()), method_name, "test"])

    np.savez(os.path.join(output_dir, "scores.npz"),
             image_ids=np.array(image_ids), scores=fused_scores, class_indices=class_indices)

    runtime = {
        "method_name": method_name,
        "inference_time_seconds": round(inference_time_seconds, 2),
        "num_test_images": int(len(image_ids)),
        "num_images_cropped": num_cropped,
        "crop_rate": round(num_cropped / len(image_ids), 4) if image_ids else 0.0,
    }
    with open(os.path.join(output_dir, "runtime.json"), "w") as f:
        json.dump(runtime, f, indent=2)

    metrics = {
        "stage1_only_top1_accuracy": stage1_acc,
        "fused_top1_accuracy": fused_acc,
        "delta_vs_stage1_only": fused_acc - stage1_acc,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nStage-1-only accuracy : {stage1_acc:.4f}")
    print(f"Fused accuracy        : {fused_acc:.4f}  (delta: {fused_acc - stage1_acc:+.4f})")
    print(f"Cropped {num_cropped}/{len(image_ids)} images ({runtime['crop_rate']*100:.1f}%)")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
