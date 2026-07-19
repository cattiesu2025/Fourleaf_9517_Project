# C Scratch ResNet18 Results Summary

This note summarises the C-role scratch CNN experiments so that B/D/E can check method names, output files, and headline results.

## Experiment Setup

- Model: ResNet18 trained from scratch (`weights=None`)
- Task: 500-class iNaturalist subset classification
- Input: RGB images resized/cropped to `224 x 224`
- Labels: `class_idx` from `data/metadata/*.csv`
- Seed: `9517`
- Main training loss: cross-entropy
- Main ablation optimizer: AdamW, learning rate `0.001`, weight decay `0.0001`
- Output contract: each method writes `predictions.csv`, `scores.npz`, `runtime.json`, and `training_history.csv`

## Main Augmentation Ablation

These three runs are the planned augmentation ablation and should be used as the main C ablation comparison.

| Method | Transform | Epochs | Optimizer | Best Epoch | Best Val Acc | Test Top-1 | Test Top-5 | Test Macro F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `resnet18_scratch_no_aug` | no random augmentation | 30 | AdamW | 23 | 22.98% | 22.62% | 44.38% | 22.52% |
| `resnet18_scratch_basic_aug` | random resized crop + horizontal flip | 30 | AdamW | 29 | 33.46% | 31.76% | 57.42% | 31.44% |
| `resnet18_scratch_strong_aug` | basic augmentations + colour jitter + rotation + RandAugment + random erasing | 30 | AdamW | 21 | 27.56% | 25.88% | 51.00% | 24.52% |

Main observation: `basic_aug` performed best among the three planned augmentation settings. `strong_aug` improved over `no_aug`, but underperformed `basic_aug`, suggesting that overly aggressive transformations can distort fine-grained species cues such as colour, texture, and local morphology.

## Additional Training Tuning

These runs are not part of the original three-way augmentation ablation. They are additional tuning/extension experiments using the best augmentation setting.

| Method | Change from Main Basic Aug | Epochs | Optimizer | Best Epoch | Best Val Acc | Test Top-1 | Test Top-5 | Test Macro F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `resnet18_scratch_basic_aug_100epoch` | longer training | 100 | AdamW | 97 | 35.84% | 34.22% | 57.44% | 33.72% |
| `resnet18_scratch_basic_aug_sgd` | longer training + SGD optimizer | 100 | SGD | 91 | 39.10% | 37.94% | 61.24% | 37.51% |

Best scratch model from the C experiments: `resnet18_scratch_basic_aug_sgd`.

## Handoff To E

The handoff archive is stored outside GitHub and should be shared via Drive:

```text
scratch_outputs_no_checkpoints.tar.gz
```

It should include:

```text
outputs/scratch/resnet18_scratch_no_aug/
outputs/scratch/resnet18_scratch_basic_aug/
outputs/scratch/resnet18_scratch_strong_aug/
outputs/scratch/resnet18_scratch_basic_aug_100epoch/
outputs/scratch/resnet18_scratch_basic_aug_sgd/
outputs/scratch/scratch_ablation_summary.csv
```

Each method directory should contain:

```text
predictions.csv
scores.npz
runtime.json
training_history.csv
```

Checkpoints (`*.pth`) are intentionally excluded from the E handoff archive and must not be pushed to GitHub.

## Suggested Report Wording

The scratch ResNet18 baseline achieved modest but meaningful performance on the 500-way fine-grained species classification task. Basic augmentation improved Top-1 accuracy from 22.62% without augmentation to 31.76%, demonstrating that simple label-preserving transformations substantially improve generalisation. Strong augmentation reached 25.88% Top-1 accuracy, outperforming no augmentation but falling below basic augmentation, likely because aggressive colour, rotation, crop, and erasing operations can disrupt fine-grained visual cues. Extending the best basic augmentation setting to 100 epochs improved Top-1 accuracy to 34.22%, and switching to SGD further improved the best scratch result to 37.94%.
