# COMP9517 Fourleaf Project

This repository contains the shared code for the COMP9517 group project,
including the scratch CNN pipeline and the Evaluation and Robustness framework.

## Evaluation and Robustness MVP

### Included deliverables

- `src/evaluation/metrics.py`
  - Top-1 accuracy
  - Top-5 accuracy
  - Overall accuracy
  - Macro precision
  - Macro recall
  - Macro F1
  - Balanced accuracy
- `src/evaluation/evaluate.py`
  - Reads `predictions.csv` and `scores.npz`
  - Aligns rows by `image_id`
  - Validates score shape and `class_indices`
  - Writes `metrics.json`
  - Calls the confusion-matrix output functions
- `src/evaluation/confusion_matrix.py`
  - Saves the full confusion matrix as `.npy` and `.png`
- `src/evaluation/degradation.py`
  - Gaussian noise
  - Gaussian blur
  - Brightness reduction
  - JPEG compression
- `configs/robustness.yaml`
  - Shared severity levels and parameter values
- `src/evaluation/robustness.py`
  - Shared `ModelPredictor` protocol
  - Model-independent robustness inference runner
  - Robustness result validation and aggregation
- `demo/evaluation_mvp/`
  - Sample `predictions.csv`
  - Sample `scores.npz`
  - Expected outputs
  - Dummy predictor
  - Run instructions

Final cross-method figures and real-model robustness results are intentionally
out of scope for this MVP. They can be generated after the final model outputs
are available.

### Evaluation environment setup

Python 3.11 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Alternatively, use the provided `uv` bootstrap script:

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
```

### Run the evaluation demo

```bash
python -m src.evaluation.evaluate \
  --predictions demo/evaluation_mvp/input/predictions.csv \
  --scores demo/evaluation_mvp/input/scores.npz \
  --runtime demo/evaluation_mvp/input/runtime.json \
  --classes demo/evaluation_mvp/input/idx_to_class.json \
  --output tmp/mvp-evaluation-output \
  --expected-num-classes 3
```

The generated metric values and confusion matrix should match the files in
`demo/evaluation_mvp/expected_output/`.

### Run the dummy-predictor robustness demo

```bash
python demo/evaluation_mvp/dummy_robustness.py \
  --output tmp/mvp-dummy-robustness \
  --force
```

This command runs one dummy predictor through four degradation types and five
severity levels. It creates 20 standard result directories and an aggregated
robustness summary.

### Run the evaluation tests

```bash
python -m pytest
```

The test suite covers metric calculation, `image_id` alignment, permuted
`class_indices`, confusion-matrix outputs, deterministic degradation, the
shared predictor interface, and the complete 4x5 robustness matrix.

## Scratch CNN Pipeline

This repository contains the C-role scratch CNN pipeline for the COMP9517 group project. It trains ResNet18 from scratch on the fixed 500-class iNaturalist subset and writes outputs in the shared evaluation format.

### Data Layout

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

### Training environment

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

### Smoke Test

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

### Train Scratch ResNet18

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

### Predict

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
