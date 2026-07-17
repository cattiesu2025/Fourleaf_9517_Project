# 开发日志 — C Scratch ResNet18
> 创建时间：2026-07-17 | 最后更新：2026-07-17
> 关联实现指南：C_implementation_plan.md
> 本文件只追加，不删除。每一次代码修改都必须追加新的日志条目。

## 项目概览

| 项目 | 内容 |
|---|---|
| 任务 | C 角色：ResNet18 from scratch + data augmentation ablation |
| 框架 | PyTorch / torchvision |
| 数据合同 | `data/metadata/*.csv` + `data/raw/` |
| 输出合同 | `outputs/scratch/<method_name>/predictions.csv`, `scores.npz`, `runtime.json`, `training_history.csv` |

## 实现进度

| 模块 | 文件 | 状态 | 完成时间 | 备注 |
|---|---|---|---|---|
| 数据加载 | `src/data/dataset.py`, `src/data/dataloader.py` | ✅ Done | 2026-07-17 | 按 metadata CSV 读取 |
| Transform | `src/data/transforms.py` | ✅ Done | 2026-07-17 | 支持 none/basic/strong |
| Longtail sampler | `src/data/sampling.py` | ✅ Done | 2026-07-17 | WeightedRandomSampler |
| Scratch model | `src/scratch/model.py` | ✅ Done | 2026-07-17 | `weights=None` |
| 训练脚本 | `src/scratch/train.py` | ✅ Done | 2026-07-17 | 输出 history/checkpoints/runtime |
| 预测脚本 | `src/scratch/predict.py` | ✅ Done | 2026-07-17 | 输出 predictions/scores/runtime |
| 配置 | `configs/scratch.yaml` | ✅ Done | 2026-07-17 | 记录五组实验 |
| README | `README.md` | 🔄 WIP | — | 待验证后补充最终说明 |

## 开发日志

### 2026-07-17 — 初始化 C pipeline
- **完成内容**：新增数据读取、augmentation、longtail sampler、scratch ResNet18、训练脚本、预测脚本和实验配置。
- **遇到的问题**：项目原先 metadata 位于根目录，与组内规范不一致。
- **解决方案**：已将 metadata 移入 `data/metadata/`，CSV 内部 `image_path` 保持 `data/raw/...`，路径检查通过。

### 2026-07-17 — Smoke test 后清理 AMP 警告
- **完成内容**：将训练脚本中的 AMP 调用更新为 `torch.amp.GradScaler` 和 `torch.amp.autocast`。
- **遇到的问题**：CPU smoke test 可运行，但旧版 `torch.cuda.amp` API 会输出弃用提醒。
- **解决方案**：改用 PyTorch 2.x 推荐接口，保持默认不开启 AMP，仅在 CUDA 且传入 `--amp` 时启用。

### 2026-07-17 — 修正 B/C/D 统一环境说明
- **完成内容**：更新 README 和数据输出规范中的 runtime 环境说明。
- **遇到的问题**：原说明只强调 C/D 深度学习实验统一 GPU 环境，容易误解为 B 可以使用不同环境。
- **解决方案**：统一改为 B/C/D 的正式训练、测试推理和 runtime 统计都在同一个环境下完成；B 的传统方法即使不使用 GPU，也应在同一平台上运行并记录硬件。

### 2026-07-17 — 增加 Katana 运行说明
- **完成内容**：将正式 runtime 环境示例扩展为 Katana 或 Colab，并在 README 中加入 Katana GPU interactive job 示例。
- **遇到的问题**：只写 Colab T4 会让人误以为必须付费使用 Colab，并且需要额外上传数据。
- **解决方案**：明确 Katana 可以作为正式统一环境；在 Katana 上需申请 compute/GPU node 后再运行 PyTorch 训练，不在 login node 上训练。

### 2026-07-17 — 增强 runtime 硬件记录
- **完成内容**：`runtime.json` 的 `hardware` 字段新增 `hostname`、`pbs_jobid`、`gpu_memory_gb` 和 `ram_gb` 自动记录。
- **遇到的问题**：Katana 正式实验需要说明具体分配到的节点和资源，仅记录平台名称不够清楚。
- **解决方案**：训练和预测共用 `hardware_info()`，在 Katana PBS job 中会自动捕获 `PBS_JOBID`。

## 运行说明

### Smoke test

```bash
python src/scratch/train.py \
  --method_name smoke_resnet18_scratch \
  --transform_type none \
  --train_csv data/metadata/train.csv \
  --val_csv data/metadata/val.csv \
  --output_dir outputs/scratch/smoke_resnet18_scratch \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 0 \
  --device cpu \
  --max_train_batches 1 \
  --max_val_batches 1
```

这条命令只跑一个训练 batch 和一个验证 batch，用于确认 dataloader、model、loss、checkpoint、history 和 runtime 能正常工作。

### Train augmentation ablation

```bash
python src/scratch/train.py \
  --method_name resnet18_scratch_basic_aug \
  --transform_type basic_aug \
  --train_csv data/metadata/train.csv \
  --val_csv data/metadata/val.csv \
  --output_dir outputs/scratch/resnet18_scratch_basic_aug \
  --epochs 30 \
  --batch_size 64 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --seed 9517
```

将 `method_name` / `transform_type` / `output_dir` 改为 `resnet18_scratch_no_aug` + `none` 或 `resnet18_scratch_strong_aug` + `strong_aug` 即可跑三组 augmentation ablation。

### Train longtail experiment

```bash
python src/scratch/train.py \
  --method_name resnet18_scratch_longtail_resampled \
  --transform_type basic_aug \
  --train_csv data/metadata/longtail_train.csv \
  --val_csv data/metadata/val.csv \
  --output_dir outputs/scratch/resnet18_scratch_longtail_resampled \
  --sampler weighted_random
```

### Predict test set

```bash
python src/scratch/predict.py \
  --method_name resnet18_scratch_basic_aug \
  --checkpoint outputs/scratch/resnet18_scratch_basic_aug/checkpoint_best.pth \
  --test_csv data/metadata/test.csv \
  --output_dir outputs/scratch/resnet18_scratch_basic_aug \
  --batch_size 64
```

输出 `predictions.csv`、`scores.npz` 和更新后的 `runtime.json`，供 E 的 evaluation pipeline 读取。
