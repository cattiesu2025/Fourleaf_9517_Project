"""
src/transfer/confusable_pairs_analysis.py
--------------------------------------------
Systematically finds "confusable species pairs" (same genus) using A's
idx_to_class.json, ranks them by how often the model actually confuses
them (from predictions.csv), and generates Grad-CAM visualizations for
the top few pairs. This completes the "next step" gradcam_analysis.py
prints at the end of its run.

Genus is taken as the first word of class_name (e.g. "Cota tinctoria"
-> genus "Cota"). Two classes are a genus pair if they share a genus
AND both were among the 500 selected classes.

Usage:
    python src/transfer/confusable_pairs_analysis.py --strategy finetuned --top_n_pairs 3

    # for the regularization/attention ablations:
    python src/transfer/confusable_pairs_analysis.py --strategy finetuned \
        --dropout_rate 0.4 --output_dir outputs/transfer/resnet18_pretrained_finetuned_regularized_extra
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.transfer.model import build_model, STRATEGY_TO_METHOD_NAME
from src.transfer.gradcam_analysis import load_predictions, visualize_examples
from src.data.dataloader import get_dataloader


def build_genus_groups(idx_to_class_path: str) -> dict[str, list[str]]:
    """Returns {genus: [class_idx, class_idx, ...]} for genera with >=2 classes."""
    with open(idx_to_class_path) as f:
        idx_to_class = json.load(f)

    genus_to_classes = defaultdict(list)
    for class_idx, info in idx_to_class.items():
        genus = info["class_name"].split()[0]
        genus_to_classes[genus].append(class_idx)

    return {genus: classes for genus, classes in genus_to_classes.items() if len(classes) >= 2}


def find_confused_pairs(pred_rows: list[dict], genus_groups: dict[str, list[str]]):
    """Counts how often the model confuses each same-genus class pair.

    Returns a list of (genus, class_a, class_b, confusion_count, example_image_ids)
    sorted by confusion_count descending.
    """
    # class_idx -> genus, for fast lookup
    class_to_genus = {}
    for genus, classes in genus_groups.items():
        for c in classes:
            class_to_genus[c] = genus

    pair_counts = defaultdict(int)
    pair_examples = defaultdict(list)

    for row in pred_rows:
        true_c, pred_c = row["true_label"], row["pred_label"]
        if true_c == pred_c:
            continue
        if true_c not in class_to_genus or pred_c not in class_to_genus:
            continue
        if class_to_genus[true_c] != class_to_genus[pred_c]:
            continue  # only same-genus confusions count here

        pair_key = tuple(sorted([true_c, pred_c]))
        pair_counts[pair_key] += 1
        pair_examples[pair_key].append(row["image_id"])

    ranked = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    results = []
    for (class_a, class_b), count in ranked:
        results.append({
            "genus": class_to_genus[class_a],
            "class_a": class_a,
            "class_b": class_b,
            "confusion_count": count,
            "example_image_ids": pair_examples[(class_a, class_b)],
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["frozen", "finetuned", "layer4"], required=True)
    parser.add_argument("--idx_to_class", default="data/metadata/idx_to_class.json")
    parser.add_argument("--test_csv", default="data/metadata/test.csv")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--output_root", default="outputs/transfer")
    parser.add_argument("--output_dir", default=None,
                         help="Exact directory to read predictions.csv/checkpoint from -- "
                              "overrides the official path, for regularized/attention runs.")
    parser.add_argument("--use_attention", action="store_true", default=False)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--top_n_pairs", type=int, default=3)
    parser.add_argument("--examples_per_pair", type=int, default=4)
    args = parser.parse_args()

    if args.output_dir:
        out_dir = args.output_dir
    else:
        method_name = STRATEGY_TO_METHOD_NAME[args.strategy]
        if args.use_attention:
            method_name = f"{method_name}_attention_extra"
        out_dir = os.path.join(args.output_root, method_name)

    pred_path = os.path.join(out_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Run predict.py first -- {pred_path} not found.")

    print("Building genus groups from idx_to_class.json...")
    genus_groups = build_genus_groups(args.idx_to_class)
    print(f"Found {len(genus_groups)} genera with >=2 classes among the selected 500.")

    pred_rows = load_predictions(pred_path)
    confused_pairs = find_confused_pairs(pred_rows, genus_groups)

    if not confused_pairs:
        print("No same-genus confusions found in predictions.csv -- nothing to visualize.")
        print("(This can happen if very few genera have multiple selected species, or if")
        print(" the model made no same-genus errors on the test set.)")
        return

    print(f"\nTop {min(args.top_n_pairs, len(confused_pairs))} same-genus confused pairs:")
    for p in confused_pairs[:args.top_n_pairs]:
        print(f"  genus={p['genus']}  class {p['class_a']} <-> class {p['class_b']}  "
              f"confused {p['confusion_count']}x")

    # Save the full ranked table for the report/appendix
    summary_path = os.path.join(out_dir, "confusable_genus_pairs.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["genus", "class_a", "class_b", "confusion_count", "example_image_ids"])
        for p in confused_pairs:
            writer.writerow([p["genus"], p["class_a"], p["class_b"], p["confusion_count"],
                              ";".join(p["example_image_ids"])])
    print(f"\nWrote full ranked table to {summary_path}")

    # --- Load model once, generate Grad-CAM for the top N pairs ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(out_dir, "checkpoint_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["num_classes"], pretrained=False,
                         use_attention=args.use_attention, num_heads=args.num_heads,
                         dropout_rate=args.dropout_rate).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loader = get_dataloader(
        split="test", csv_file=args.test_csv, transform_type="none",
        batch_size=64, shuffle=False, image_size=args.image_size, seed=args.seed,
    )

    fig_dir = os.path.join(out_dir, "gradcam_examples", "confusable_pairs")
    os.makedirs(fig_dir, exist_ok=True)

    for p in confused_pairs[:args.top_n_pairs]:
        target_ids = set(p["example_image_ids"][:args.examples_per_pair])
        out_path = os.path.join(
            fig_dir, f"genus_{p['genus']}_class{p['class_a']}_vs_class{p['class_b']}.png"
        )
        title = (f"Grad-CAM: confused pair (genus={p['genus']}, "
                 f"class {p['class_a']} vs {p['class_b']}, {p['confusion_count']}x confused)")
        visualize_examples(model, test_loader, target_ids, device, out_path, title)


if __name__ == "__main__":
    main()
