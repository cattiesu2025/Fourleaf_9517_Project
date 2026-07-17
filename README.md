# COMP9517 Fourleaf Project

This repository contains the C-role scratch CNN pipeline for the COMP9517 group project. It trains ResNet18 from scratch on the fixed 500-class iNaturalist subset and writes outputs in the shared evaluation format.

## Data Layout

Required local files:

```text
data/
├── raw/
│   ├── train_mini/
│   └── val/
└── metadata/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── longtail_train.csv
    ├── class_to_idx.json
    └── idx_to_class.json
```

`data/raw/` is gitignored. `data/metadata/` should be tracked because it defines the fixed split and class mapping.

## Environment

Use Python 3.11 and install the project dependencies:

```bash
pip install -r requirements.txt
```

For formal runtime comparisons, run the final B/C/D training and inference jobs in the same environment, such as Katana jobs with the same resource request or the same Colab T4 runtime, and record that hardware in `runtime.json`. Traditional B methods may not use the GPU, but they should still run on the same platform for fair timing records.

Katana is a good option when the dataset is already on UNSW storage or can be copied once to shared scratch. Request a GPU compute node before running PyTorch training; do not train on the login node. A typical interactive request is:

```bash
qsub -I -l select=1:ncpus=8:ngpus=1:mem=46gb,walltime=2:00:00
```

After entering the allocated compute node, run the training command with CUDA:

```bash
python src/scratch/train.py \
  --method_name resnet18_scratch_basic_aug \
  --transform_type basic_aug \
  --train_csv data/metadata/train.csv \
  --val_csv data/metadata/val.csv \
  --output_dir outputs/scratch/resnet18_scratch_basic_aug \
  --epochs 30 \
  --batch_size 64 \
  --device cuda \
  --amp \
  --seed 9517
```

## Smoke Test

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

## Train Scratch ResNet18

Basic augmentation baseline:

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

No augmentation:

```bash
python src/scratch/train.py \
  --method_name resnet18_scratch_no_aug \
  --transform_type none \
  --train_csv data/metadata/train.csv \
  --val_csv data/metadata/val.csv \
  --output_dir outputs/scratch/resnet18_scratch_no_aug \
  --epochs 30 \
  --batch_size 64 \
  --seed 9517
```

Strong augmentation:

```bash
python src/scratch/train.py \
  --method_name resnet18_scratch_strong_aug \
  --transform_type strong_aug \
  --train_csv data/metadata/train.csv \
  --val_csv data/metadata/val.csv \
  --output_dir outputs/scratch/resnet18_scratch_strong_aug \
  --epochs 30 \
  --batch_size 64 \
  --seed 9517
```

Longtail with dynamic weighted sampling:

```bash
python src/scratch/train.py \
  --method_name resnet18_scratch_longtail_resampled \
  --transform_type basic_aug \
  --train_csv data/metadata/longtail_train.csv \
  --val_csv data/metadata/val.csv \
  --output_dir outputs/scratch/resnet18_scratch_longtail_resampled \
  --sampler weighted_random \
  --epochs 30 \
  --batch_size 64 \
  --seed 9517
```

## Predict

```bash
python src/scratch/predict.py \
  --method_name resnet18_scratch_basic_aug \
  --checkpoint outputs/scratch/resnet18_scratch_basic_aug/checkpoint_best.pth \
  --test_csv data/metadata/test.csv \
  --output_dir outputs/scratch/resnet18_scratch_basic_aug \
  --batch_size 64
```

Each method directory writes:

```text
predictions.csv
scores.npz
training_history.csv
runtime.json
checkpoint_best.pth
checkpoint_last.pth
```

Checkpoints and `outputs/` are gitignored and should not be included in the final code ZIP.
