# 运行指令

## 1. 训练(frozen + finetuned 都跑)

```bash
python src/transfer/train.py --strategy frozen --epochs 15 --pretrained
python src/transfer/train.py --strategy finetuned --epochs 15 --pretrained
```

## 2. 预测

```bash
python src/transfer/predict.py --strategy frozen
python src/transfer/predict.py --strategy finetuned
```

## 3. 算指标(top-1/top-5/macro-F1,两个方法对比表)

```bash
python src/transfer/compute_metrics.py --compare
```

## 4. 训练曲线对比图

```bash
python src/transfer/plot_training_curves.py
```

## 5. Grad-CAM可视化(正确/错误分类案例)

```bash
python src/transfer/gradcam_analysis.py --strategy frozen --num_examples 6
python src/transfer/gradcam_analysis.py --strategy finetuned --num_examples 6
```

## 6. Frozen vs Finetuned 同一张图的Grad-CAM对比

```bash
python src/transfer/compare_gradcam_models.py --image_id 50
```

## 7. 系统性同属混淆物种对分析

```bash
python src/transfer/confusable_pairs_analysis.py --strategy finetuned --top_n_pairs 3
```

## 8. Grad-CAM二阶段裁剪融合（Command 1 为测试）

```bash
python src/transfer/gradcam_crop_fusion.py --strategy finetuned --max_images 20
python src/transfer/gradcam_crop_fusion.py --strategy finetuned
```

---

所有结果都在 `outputs/transfer/` 下对应的方法文件夹里。
