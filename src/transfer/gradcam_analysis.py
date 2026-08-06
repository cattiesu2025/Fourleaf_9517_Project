"""Generate Grad-CAM analysis figures for transfer-learning runs:

    - correctly vs incorrectly classified examples
    - confusable species pairs (e.g. same genus)

Reads outputs/transfer/<method_name>/predictions.csv (already produced by
predict.py) to pick which test images to visualize, so it doesn't re-run
inference from scratch.

Output: outputs/transfer/<method_name>/gradcam_examples/*.png

Usage:
    ./scripts/comp9517 transfer-gradcam --strategy finetuned --num-examples 6
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.cli import ArgumentParser
from src.data.dataloader import get_dataloader
from src.transfer.gradcam import GradCAM, denormalize_image, overlay_heatmap
from src.transfer.model import STRATEGY_TO_METHOD_NAME, build_model


def load_predictions(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_examples(pred_rows, num_correct, num_incorrect):
    correct = [r for r in pred_rows if r["true_label"] == r["pred_label"]][:num_correct]
    incorrect = [r for r in pred_rows if r["true_label"] != r["pred_label"]][:num_incorrect]
    return correct, incorrect


def visualize_examples(model, loader, target_ids, device, out_path, title):
    """target_ids: set of image_id strings (as read from predictions.csv) to look for.

    The dataset parser converts numeric image_id values to
    int, so the DataLoader batches them into a torch.LongTensor -- iterating it
    yields tensor scalars, not plain ints/strs. Normalize everything to str
    before comparing against target_ids (which come from predictions.csv as
    strings), or every lookup silently fails to match.
    """
    id_to_batch_item = {}
    for images, labels, image_ids in loader:
        image_ids = image_ids.tolist() if hasattr(image_ids, "tolist") else list(image_ids)
        for i, raw_iid in enumerate(image_ids):
            iid = str(raw_iid)
            if iid in target_ids:
                id_to_batch_item[iid] = (images[i : i + 1], labels[i].item())
        if len(id_to_batch_item) == len(target_ids):
            break

    n = len(id_to_batch_item)
    if n == 0:
        print(f"No matching examples found for {title}; skipping.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    cam = GradCAM(model, target_layer=model.layer4[-1])
    for row_idx, (iid, (img_tensor, label)) in enumerate(id_to_batch_item.items()):
        img_tensor = img_tensor.to(device)
        heatmap, pred_class = cam.generate(img_tensor, class_idx=None)
        img_uint8 = denormalize_image(img_tensor[0])
        overlaid = overlay_heatmap(img_uint8, heatmap, alpha=0.4)

        axes[row_idx, 0].imshow(img_uint8)
        axes[row_idx, 0].set_title(f"{iid} (true={label})")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(overlaid)
        axes[row_idx, 1].set_title(f"Grad-CAM (pred={pred_class})")
        axes[row_idx, 1].axis("off")
    cam.remove_hooks()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["frozen", "finetuned", "layer4"], required=True)
    parser.add_argument("--num_examples", type=int, default=6)
    parser.add_argument("--test_csv", default="data/metadata/test.csv")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--output_root", default="outputs/transfer")
    parser.add_argument("--use_attention", action="store_true", default=False)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="Must match the value used in train.py for this checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Exact directory to read predictions.csv/checkpoint from and "
        "write gradcam_examples/ into; overrides the standard "
        "outputs/transfer/<method_name>/ path. Use this for the "
        "regularization ablation (e.g. .../resnet18_pretrained_finetuned_regularized_extra).",
    )
    args = parser.parse_args()

    if args.output_dir:
        out_dir = args.output_dir
    else:
        method_name = STRATEGY_TO_METHOD_NAME[args.strategy]
        if args.use_attention:
            method_name = f"{method_name}_attention_extra"
        out_dir = os.path.join(args.output_root, method_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_path = os.path.join(out_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Run predict.py first -- {pred_path} not found.")

    pred_rows = load_predictions(pred_path)
    correct, incorrect = find_examples(pred_rows, args.num_examples // 2, args.num_examples // 2)

    # checkpoint_best.pth, matching train.py's naming (not the old checkpoints/transfer/model.pt)
    ckpt_path = os.path.join(out_dir, "checkpoint_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(
        ckpt["num_classes"],
        pretrained=False,
        use_attention=args.use_attention,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loader = get_dataloader(
        split="test",
        csv_file=args.test_csv,
        transform_type="none",
        batch_size=64,
        shuffle=False,
        image_size=args.image_size,
        seed=args.seed,
    )

    fig_dir = os.path.join(out_dir, "gradcam_examples")
    os.makedirs(fig_dir, exist_ok=True)

    correct_ids = {r["image_id"] for r in correct}
    incorrect_ids = {r["image_id"] for r in incorrect}

    visualize_examples(
        model,
        test_loader,
        correct_ids,
        device,
        os.path.join(fig_dir, "correct_examples.png"),
        "Grad-CAM: correctly classified examples",
    )
    visualize_examples(
        model,
        test_loader,
        incorrect_ids,
        device,
        os.path.join(fig_dir, "incorrect_examples.png"),
        "Grad-CAM: misclassified examples",
    )

    print("\nNext step for confusable species pairs (e.g. same genus):")
    print("  Use idx_to_class.json to look up which class_idx values")
    print("  share a genus/category, filter predictions.csv for those class_idx pairs,")
    print("  and call visualize_examples() again with those image_ids.")


if __name__ == "__main__":
    main()
