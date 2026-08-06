# COMP9517 Fourleaf Project

This repository contains the implementation for the COMP9517 group project,
including the scratch CNN pipeline and the Evaluation and Robustness framework.

## Quick start

Python 3.11 is required. The recommended setup uses `uv` to create the
environment and install the pinned dependencies:

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
./scripts/comp9517 check-environment
./scripts/comp9517 test
```

Without `uv`, create a Python 3.11 environment manually:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Run `./scripts/comp9517 --help` to see every data, training, prediction,
evaluation, robustness, and analysis command. Command details are
documented in [`docs/commands.md`](docs/commands.md).

Raw iNaturalist images and trained checkpoints are intentionally not included.
Place downloaded images under `data/raw/` and model artifacts under `outputs/`
when running workflows that need them.

## Third-party libraries and external resources

The repository does not vendor or copy third-party library source code. Project
modules call the pinned packages below through their public Python APIs; the
packages themselves are installed from `requirements.txt` and are not vendored
in this repository.

| Dependency | How it is used |
| --- | --- |
| [PyTorch](https://pytorch.org/) and [Torchvision](https://pytorch.org/vision/stable/) | Neural-network training and inference, ResNet18 architecture, image transforms, and optional ImageNet initialization. |
| [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [SciPy](https://scipy.org/) | Numerical arrays, metadata tables, result serialization, and scientific utilities. |
| [scikit-learn](https://scikit-learn.org/stable/) | Random Forest, MiniBatchKMeans, Linear SVM, and evaluation metrics. |
| [OpenCV](https://opencv.org/) and [scikit-image](https://scikit-image.org/) | SIFT descriptors, image operations, HOG features, and quality checks. |
| [Pillow](https://python-pillow.org/) | Image loading and deterministic degradation operations. |
| [Matplotlib](https://matplotlib.org/) | Confusion matrices, comparison plots, and Grad-CAM figures written to generated-output directories. |
| [PyYAML](https://github.com/yaml/pyyaml) | Experiment and robustness configuration loading. |
| [tqdm](https://tqdm.github.io/) and [ijson](https://github.com/ICRAR/ijson) | Progress reporting and streaming the large official metadata JSON. |

Development-only dependencies are `pytest` for automated tests and `ruff` for
linting and formatting. The raw
[iNaturalist 2021](https://github.com/visipedia/inat_comp/tree/master/2021)
dataset is an external resource and is never bundled. When transfer learning is
run with `--pretrained`, Torchvision obtains
`ResNet18_Weights.IMAGENET1K_V1`; those ImageNet weights and all trained project
checkpoints are stored outside Git. The repository retains the fixed CSV/JSON
split metadata and the project-specific Python, shell, and YAML files.

## Repository structure

```text
src/common/       Shared CLI, runtime, and prediction helpers
src/data/         Metadata-driven datasets, transforms, and sampling
src/traditional/  HOG/Random Forest and SIFT/BoVW/SVM methods
src/scratch/      Scratch-trained ResNet18 workflow
src/transfer/     Transfer-learning and Grad-CAM workflows
src/evaluation/   Output validation, metrics, comparison, and robustness
scripts/          Reproducible data and experiment commands
configs/          Versioned experiment configuration
tests/            Automated regression tests
data/metadata/    Fixed non-image split and class metadata
docs/             Public run documentation
```

Module docstrings and inline comments document non-obvious behavior such as
score alignment, feature-cache keys, deterministic degradation, and artifact
contracts. Generated data and results remain outside the source modules.

### Rebuild dataset metadata

The fixed split files in `data/metadata/` are versioned. Rebuild them only
when intentionally creating a new split from the official annotations:

```bash
./scripts/comp9517 build-split --force
./scripts/comp9517 build-longtail --force
```

To inspect or copy only the images referenced by the fixed train, validation,
and test CSV files:

```bash
./scripts/comp9517 copy-selected-images --dry-run
./scripts/comp9517 copy-selected-images --output-dir data/subset
```

## Evaluation and Robustness MVP

### Included components

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
- `src/evaluation/compare.py`
  - Revalidates every reference prediction and score artifact
  - Enforces one shared 5,000-image test mapping
  - Generates the final cross-method tables and report figures
- `src/evaluation/model_adapters.py` and `scripts/run_final_robustness.py`
  - Load the HOG, SIFT, scratch, and fine-tuned model artifacts
  - Reproduce reference clean predictions before degraded-image inference
  - Run and aggregate the final four-degradation by five-severity matrix
- `demo/evaluation_mvp/`
  - Sample `predictions.csv`
  - Sample `scores.npz`
  - Expected outputs
  - Dummy predictor
  - Run instructions

The unified cross-method tables, 80-run real-model robustness summary, and
paper figures are generated under `report/generated/`.  Re-running robustness
requires the gitignored original test images under `data/raw/val/`.

### Evaluation environment setup

Use the Python 3.11 environment created in [Quick start](#quick-start), then
run `./scripts/comp9517 check-environment` before producing formal results.

### Unified project command

Use one entry point for data preparation, training, prediction, evaluation,
robustness, and tests:

```bash
./scripts/comp9517 --help
./scripts/comp9517 <command> --help
```

New examples use kebab-case options such as `--batch-size`. The previous
snake_case spellings remain accepted, so saved experiment commands do not
break. See [`docs/commands.md`](docs/commands.md) for the complete command map.

### Run the evaluation demo

```bash
./scripts/comp9517 evaluate \
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
./scripts/comp9517 demo-robustness \
  --output tmp/mvp-dummy-robustness \
  --force
```

This command runs one dummy predictor through four degradation types and five
severity levels. It creates 20 standard result directories and an aggregated
robustness summary.

### Run the evaluation tests

```bash
./scripts/comp9517 test
```

The test suite covers metric calculation, `image_id` alignment, permuted
`class_indices`, confusion-matrix outputs, deterministic degradation, the
shared predictor interface, and the complete 4x5 robustness matrix.

### Generate the final cross-method report artifacts

After placing the trained artifacts in a local artifact directory, run:

```bash
./scripts/comp9517 compare \
  --manifest configs/final_evaluation.yaml \
  --artifact-root outputs/model_artifacts \
  --output report/generated
```

The command refuses to generate tables if any run has malformed scores or a
different `image_id -> true_label` mapping.

### Run final real-model robustness

The fixed test CSV points to 5,000 images from the official iNaturalist 2021
validation archive.  Once those files exist under `data/raw/val/`, run:

```bash
./scripts/comp9517 run-robustness \
  --artifact-root outputs/model_artifacts \
  --test-csv data/metadata/test.csv \
  --data-root . \
  --output outputs/robustness \
  --evaluation-output outputs/evaluation/robustness \
  --device auto \
  --batch-size 64 \
  --clean-check-images 5000 \
  --resume
```

Before the first degraded run for each model, the command records an
undegraded severity-0 baseline in the same environment and checks it against
the reference clean predictions.  The final summary calculates absolute
Top-1/Macro-F1 drops from this local baseline.  This avoids treating small
x86/ARM numerical drift in handcrafted feature extraction as degradation loss;
the reference artifacts remain the source for the main comparison table.

To perform only the clean-prediction check, add `--preflight-only`.

## Scratch CNN Pipeline

The scratch CNN pipeline trains ResNet18 from scratch on the fixed 500-class
iNaturalist subset and writes outputs in the shared evaluation format.

### Data Layout

Required local files:

```text
data/
|-- raw/
|   |-- train_mini/
|   `-- val/
`-- metadata/
    |-- train.csv
    |-- val.csv
    |-- test.csv
    |-- longtail_train.csv
    |-- class_to_idx.json
    `-- idx_to_class.json
```

`data/raw/` is gitignored. `data/metadata/` should be tracked because it defines the fixed split and class mapping.

### Training environment

Use Python 3.11 and install the project dependencies:

```bash
pip install -r requirements.txt
```

The completed limited-data artifacts were produced on different hardware.
Their `runtime.json` values are retained as descriptive cost
records, not as a controlled speed ranking.  The full-data pair used the same
L40S GPU model on different physical nodes and is therefore only approximately
comparable in elapsed time.

Katana is a good option when the dataset is already on UNSW storage or can be copied once to shared scratch. Request a GPU compute node before running PyTorch training; do not train on the login node. A typical interactive request is:

```bash
qsub -I -l select=1:ncpus=8:ngpus=1:mem=46gb,walltime=2:00:00
```

After entering the allocated compute node, run the training command with CUDA:

```bash
./scripts/comp9517 train-scratch \
  --method-name resnet18_scratch_basic_aug \
  --transform-type basic_aug \
  --train-csv data/metadata/train.csv \
  --val-csv data/metadata/val.csv \
  --output-dir outputs/scratch/resnet18_scratch_basic_aug \
  --epochs 30 \
  --batch-size 64 \
  --device cuda \
  --amp \
  --seed 9517
```

### Smoke Test

```bash
./scripts/comp9517 train-scratch \
  --method-name smoke_resnet18_scratch \
  --transform-type none \
  --train-csv data/metadata/train.csv \
  --val-csv data/metadata/val.csv \
  --output-dir outputs/scratch/smoke_resnet18_scratch \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --device cpu \
  --max-train-batches 1 \
  --max-val-batches 1
```

### Train Scratch ResNet18

Basic augmentation baseline:

```bash
./scripts/comp9517 train-scratch \
  --method-name resnet18_scratch_basic_aug \
  --transform-type basic_aug \
  --train-csv data/metadata/train.csv \
  --val-csv data/metadata/val.csv \
  --output-dir outputs/scratch/resnet18_scratch_basic_aug \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --seed 9517
```

No augmentation:

```bash
./scripts/comp9517 train-scratch \
  --method-name resnet18_scratch_no_aug \
  --transform-type none \
  --train-csv data/metadata/train.csv \
  --val-csv data/metadata/val.csv \
  --output-dir outputs/scratch/resnet18_scratch_no_aug \
  --epochs 30 \
  --batch-size 64 \
  --seed 9517
```

Strong augmentation:

```bash
./scripts/comp9517 train-scratch \
  --method-name resnet18_scratch_strong_aug \
  --transform-type strong_aug \
  --train-csv data/metadata/train.csv \
  --val-csv data/metadata/val.csv \
  --output-dir outputs/scratch/resnet18_scratch_strong_aug \
  --epochs 30 \
  --batch-size 64 \
  --seed 9517
```

Longtail with dynamic weighted sampling:

```bash
./scripts/comp9517 train-scratch \
  --method-name resnet18_scratch_longtail_resampled \
  --transform-type basic_aug \
  --train-csv data/metadata/longtail_train.csv \
  --val-csv data/metadata/val.csv \
  --output-dir outputs/scratch/resnet18_scratch_longtail_resampled \
  --sampler weighted_random \
  --epochs 30 \
  --batch-size 64 \
  --seed 9517
```

### Predict

```bash
./scripts/comp9517 predict-scratch \
  --method-name resnet18_scratch_basic_aug \
  --checkpoint outputs/scratch/resnet18_scratch_basic_aug/checkpoint_best.pth \
  --test-csv data/metadata/test.csv \
  --output-dir outputs/scratch/resnet18_scratch_basic_aug \
  --batch-size 64
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

Checkpoints and `outputs/` are gitignored and remain outside the repository.
