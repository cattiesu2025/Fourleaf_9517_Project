# Transfer-learning workflow

## 1. Train frozen and fine-tuned models

```bash
./scripts/comp9517 train-transfer --strategy frozen --epochs 15 --pretrained
./scripts/comp9517 train-transfer --strategy finetuned --epochs 15 --pretrained
```

## 2. Generate predictions

```bash
./scripts/comp9517 predict-transfer --strategy frozen
./scripts/comp9517 predict-transfer --strategy finetuned
```

## 3. Compute and compare metrics

```bash
./scripts/comp9517 transfer-metrics --compare
```

## 4. Compare training curves

```bash
./scripts/comp9517 transfer-curves
```

## 5. Generate Grad-CAM examples

```bash
./scripts/comp9517 transfer-gradcam --strategy frozen --num-examples 6
./scripts/comp9517 transfer-gradcam --strategy finetuned --num-examples 6
```

## 6. Compare frozen and fine-tuned Grad-CAM results

```bash
./scripts/comp9517 transfer-gradcam-compare --image-id 50
```

## 7. Analyze same-genus confusions

```bash
./scripts/comp9517 transfer-confusions --strategy finetuned --top-n-pairs 3
```

## 8. Run two-stage Grad-CAM crop fusion

The first command is a quick check on 20 images. The second command processes
the full prediction set.

```bash
./scripts/comp9517 transfer-crop-fusion --strategy finetuned --max-images 20
./scripts/comp9517 transfer-crop-fusion --strategy finetuned
```

All generated artifacts are written to method-specific directories under
`outputs/transfer/`.
