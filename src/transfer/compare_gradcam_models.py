"""
src/transfer/compare_gradcam_models.py
------------------------------------------
Generates a single clean figure comparing Grad-CAM between two models
(e.g. frozen vs fine-tuned) on the SAME test image -- for the report's
"Frozen vs Fine-tuned: Attention Quality" figure. Regenerates Grad-CAM
fresh for both models rather than cropping the existing 6-panel
correct_examples.png/incorrect_examples.png grids, so image quality
isn't degraded.

Usage:
    python src/transfer/compare_gradcam_models.py --image_id 50

    # compare any two runs (not just the two official strategies):
    python src/transfer/compare_gradcam_models.py --image_id 50 \
        --dir_a outputs/transfer/resnet18_pretrained_frozen --label_a Frozen \
        --dir_b outputs/transfer/resnet18_pretrained_finetuned --label_b Finetuned
"""

import os
import sys
import argparse

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.transfer.model import build_model
from src.transfer.gradcam import GradCAM, overlay_heatmap, denormalize_image
from src.data.dataloader import get_dataloader


def load_model_from_dir(out_dir, device, use_attention=False, num_heads=8, dropout_rate=0.0):
    ckpt_path = os.path.join(out_dir, "checkpoint_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["num_classes"], pretrained=False,
                         use_attention=use_attention, num_heads=num_heads,
                         dropout_rate=dropout_rate).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def find_image_tensor(loader, target_image_id):
    """Scans the test loader for one specific image_id, returns (image_tensor, true_label)."""
    for images, labels, image_ids in loader:
        image_ids = image_ids.tolist() if hasattr(image_ids, "tolist") else list(image_ids)
        for i, iid in enumerate(image_ids):
            if str(iid) == str(target_image_id):
                return images[i:i+1], labels[i].item()
    raise ValueError(f"image_id {target_image_id} not found in test set.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id", required=True, help="image_id to compare, e.g. 50")
    parser.add_argument("--dir_a", default="outputs/transfer/resnet18_pretrained_frozen")
    parser.add_argument("--label_a", default="Frozen")
    parser.add_argument("--dir_b", default="outputs/transfer/resnet18_pretrained_finetuned")
    parser.add_argument("--label_b", default="Fine-tuned")
    parser.add_argument("--test_csv", default="data/metadata/test.csv")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--out_file", default="outputs/transfer/gradcam_frozen_vs_finetuned_example.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_dataloader(
        split="test", csv_file=args.test_csv, transform_type="none",
        batch_size=64, shuffle=False, image_size=args.image_size, seed=args.seed,
    )

    img_tensor, true_label = find_image_tensor(test_loader, args.image_id)
    img_tensor = img_tensor.to(device)
    img_uint8 = denormalize_image(img_tensor[0])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(img_uint8)
    axes[0].set_title(f"Original (id={args.image_id}, true={true_label})")
    axes[0].axis("off")

    for ax, out_dir, label in [(axes[1], args.dir_a, args.label_a),
                                (axes[2], args.dir_b, args.label_b)]:
        model = load_model_from_dir(out_dir, device)
        cam = GradCAM(model, target_layer=model.layer4[-1])
        heatmap, pred_class = cam.generate(img_tensor, class_idx=None)
        cam.remove_hooks()
        overlaid = overlay_heatmap(img_uint8, heatmap, alpha=0.4)

        ax.imshow(overlaid)
        ax.set_title(f"{label} (pred={pred_class})")
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    plt.savefig(args.out_file, dpi=150)
    plt.close(fig)
    print(f"Saved {args.out_file}")


if __name__ == "__main__":
    main()
