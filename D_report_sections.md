# Transfer Learning

## Experimental Setup

ResNet18 pretrained on ImageNet, two strategies: **Frozen** (only fc trained) and **Fine-tuned** (all layers trained). 15 epochs, AdamW, cosine schedule (frozen lr=1e-3, fine-tuned lr=1e-4), batch size 64, 224×224 input. Checkpoints selected by best validation accuracy.

## Results

| Method | Top-1 | Top-5 | Macro-F1 | Training time |
|---|---:|---:|---:|---:|
| C: ResNet18 scratch (best) | 37.94% | 61.24% | 37.51% | — |
| Frozen | 53.88% | 76.68% | 53.55% | 557.8s |
| **Fine-tuned** | **66.24%** | **85.82%** | **66.20%** | 725.7s |

Transfer learning substantially outperforms training from scratch; fine-tuning yields a further ~12pp gain over frozen at ~30% more training cost.

![Frozen vs Finetuned Comparison of Training Curves](outputs/transfer/training_curves_comparison.png)
*Figure 1: Comparison of training and validation loss and accuracy curves for the “Frozen” and “Fine-tuned” strategies*

Fine-tuned shows a pronounced train/val gap by the final epoch (train 98.29% vs val 66.32%, gap=31.97pp), indicating overfitting given ~40 images/class. Frozen's gap is smaller (~25pp) due to limited trainable capacity.

## Ablation: Regularization and Model Capacity

| Variant | Top-1 | Macro-F1 | Train/Val Gap | vs baseline |
|---|---:|---:|---:|---|
| Fine-tuned (baseline) | 66.24% | 66.20% | 31.97pp | — |
| + Regularization (dropout 0.4, wd 5e-4, label smoothing 0.1) | 65.20% | 65.11% | 24.43pp | Gap↓7.5pp, Top-1↓1.0pp |
| + Multi-head self-attention (post-layer4, 8 heads) | 62.08% | 62.04% | 36.18pp | Gap↑4.2pp, Top-1↓4.2pp |

![Finetuned vs Regularized Training Curve](outputs/transfer/training_curves_finetuned_vs_finetunedregularized.png)
*Figure 2: Comparison of training curves between the fine-tuned baseline and the regularized version*

![Finetuned vs Attention Model Training Curves](outputs/transfer/training_curves_finetuned_vs_finetunedattention.png)
*Figure 3: Comparison of training curves between the fine-tuned baseline and the Attention version*

Regularization (reduced effective capacity) narrows the gap with negligible test-performance cost. Adding self-attention (increased capacity, no pretrained prior) worsens both the gap and test performance — with only ~40 images/class, the randomly-initialized attention parameters cannot learn a generalizable spatial pattern and instead add memorization capacity. These two opposite-direction interventions jointly support: **model capacity must match data scale**.

---

# Advanced Method: Grad-CAM Explainability

## Method

Grad-CAM implemented from scratch (forward/backward hooks on `layer4[-1]`), following Selvaraju et al. (ICCV 2017).

## Frozen vs Fine-tuned: Attention Quality

![Frozen vs Finetuned Grad-CAM Comparison](outputs/transfer/gradcam_frozen_vs_finetuned_example.png)
*图4: 同一张图片上,Frozen(左)与Fine-tuned(右)的Grad-CAM热力图对比 —— 需要你自己把两张correct_examples.png里对应的同一张图(如#50)拼在一起，或分别插入两张图*

On identical images, frozen's activations show diffuse/off-target responses (isolated background hotspots); fine-tuned's are more tightly concentrated on the flower itself.

## Failure Case Analysis (three modes)

![Examples of Incorrect Classification](outputs/transfer/resnet18_pretrained_finetuned/gradcam_examples/incorrect_examples.png)
*图5: 三类失败模式代表案例(#51数据质量问题 / #52细粒度混淆 / #53多目标干扰)*

1. **Data quality** (#51): near-absence of the organism in frame — a labeling/capture failure.
2. **Genuine fine-grained confusion** (#52): attention correctly centered, prediction still wrong.
3. **Multi-object interference** (#53): multiple plants produce split activations.

## Cross-Model Consistency and Cross-Family Confusion

Frozen and fine-tuned produced **identical wrong predictions** on the same images (both: true=0→pred=254 on #52; true=0→pred=195 on #53).

| class_idx | Species | Family |
|---|---|---|
| 0 | *Cota tinctoria* | Asteraceae |
| 254 | *Gaillardia pinnatifida* | Asteraceae |
| 195 | *Eschscholzia caespitosa* | Papaveraceae |

0↔254 is same-family (Asteraceae) confusion — expected. 0↔195 crosses families entirely, suggesting the model's errors are **not always aligned with taxonomic relatedness**.

## Systematic Same-Genus Confusable Pairs

| Genus | Class pair | Confusions | Domain |
|---|---|---:|---|
| *Lupinus* | 127 ↔ 145 | 6x | Plants |
| *Eschscholzia* | 149 ↔ 195 | 6x | Plants |
| *Ischnura* | 160 ↔ 397 | 6x | **Insects** |

![Eschscholzia: Confusion Within the Genus](outputs/transfer/resnet18_pretrained_finetuned/gradcam_examples/confusable_pairs/genus_Eschscholzia_class149_vs_class195.png)
*图6a: Eschscholzia属内混淆(class 149 vs 195)*

![Ischnura: Confusion Within the Genus](outputs/transfer/resnet18_pretrained_finetuned/gradcam_examples/confusable_pairs/genus_Ischnura_class160_vs_class397.png)
*图6b: Ischnura属内混淆(class 160 vs 397，昆虫，证明现象不限于植物)*

![Lupinus: Confusion Within the Genus](outputs/transfer/resnet18_pretrained_finetuned/gradcam_examples/confusable_pairs/genus_Lupinus_class127_vs_class145.png)
*图7: Lupinus属内混淆(class 127 vs 145) —— 注意力集中在叶片而非花穗，与图6性质不同*

This confirms same-genus confusion is not limited to plants. We further observe two distinct attention mechanisms: *Eschscholzia*/*Ischnura* show tight, correct organism-level localization (genuine species similarity is the failure cause); *Lupinus* shows attention on leaf structure rather than the flower spike (the actual discriminative feature), suggesting genus-level cues were learned without species-level cues.

## Summary

Grad-CAM analysis shows: (1) fine-tuning sharpens attention focus; (2) failures fall into distinguishable categories; (3) misclassification is not fully explained by taxonomic distance; (4) systematic genus-level analysis confirms fine-grained confusion generalizes across plants and insects, revealing two distinct attention failure patterns.

---
