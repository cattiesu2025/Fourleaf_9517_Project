# Project command reference

Run project workflows from the repository root through one entry point:

```bash
./scripts/comp9517 --help
./scripts/comp9517 <command> --help
```

The repository-local wrapper selects `.venv/bin/python` automatically when
that environment exists.

## Environment and verification

| Command | Purpose |
| --- | --- |
| `check-environment` | Validate Python and pinned dependency versions. |
| `test` | Run the complete pytest suite. |
| `build-demo` | Regenerate the checked-in evaluation demo. |
| `demo-robustness` | Run the small 4-by-5 dummy robustness matrix. |

## Data preparation

| Command | Purpose |
| --- | --- |
| `build-split` | Build the fixed class split and metadata from official annotations. |
| `build-longtail` | Build deterministic long-tail training metadata. |
| `copy-selected-images` | Copy only the images referenced by selected split files. |
| `build-full-subset` | Build full-training-subset metadata. |
| `extract-full-subset` | Stream the official archive and extract selected images. |
| `scan-data` | Run duplicate, corruption, and image-quality checks. |

The fixed metadata checked into `data/metadata/` should only be rebuilt
deliberately. The split and long-tail builders therefore refuse to overwrite
existing files unless `--force` is supplied. To inspect the size and
availability of the selected image set without copying it, run:

```bash
./scripts/comp9517 copy-selected-images --dry-run
```

## Model workflows

| Command | Purpose |
| --- | --- |
| `train-hog` | Train or evaluate HOG + Random Forest. |
| `train-sift` | Train or evaluate SIFT + BoVW + SVM. |
| `sweep-hog` | Run HOG development configurations from YAML. |
| `sweep-sift` | Run SIFT development configurations from YAML. |
| `train-scratch` | Train scratch ResNet18. |
| `predict-scratch` | Generate scratch ResNet18 prediction artifacts. |
| `train-transfer` | Train frozen, fine-tuned, or ablation ResNet18. |
| `predict-transfer` | Generate transfer-learning prediction artifacts. |

Traditional sweep definitions live in
[`configs/traditional_sweeps.yaml`](../configs/traditional_sweeps.yaml). Inspect
or select runs without starting expensive work:

```bash
./scripts/comp9517 sweep-hog --list
./scripts/comp9517 sweep-hog --dry-run
./scripts/comp9517 sweep-hog --run trees-100
```

Sweep outputs and logs are grouped by method:

```text
outputs/traditional/<method>/dev/<run-name>/
logs/traditional/<method>/<run-name>.log
```

The root-level `run_hog_sweep.sh` and `run_sift_sweep.sh` files remain as thin
compatibility wrappers. New automation should call `comp9517 sweep-hog` or
`comp9517 sweep-sift` directly.

## Evaluation and analysis

| Command | Purpose |
| --- | --- |
| `evaluate` | Validate and evaluate one prediction artifact. |
| `compare` | Build the final cross-method tables and figures. |
| `evaluate-robustness` | Aggregate completed robustness runs. |
| `run-robustness` | Execute the configured real-model robustness matrix. |
| `transfer-metrics` | Compute transfer-learning metrics. |
| `transfer-curves` | Plot transfer-learning training curves. |
| `transfer-gradcam` | Generate Grad-CAM examples. |
| `transfer-gradcam-compare` | Compare two model Grad-CAM outputs. |
| `transfer-confusions` | Analyze confusable species pairs. |
| `transfer-crop-fusion` | Run Grad-CAM crop fusion. |

## Argument compatibility

Long options are standardized on kebab-case:

```bash
./scripts/comp9517 train-scratch --batch-size 64 --output-dir outputs/scratch/run
```

Legacy snake_case forms such as `--batch_size` and `--output_dir` are still
accepted by model commands to preserve existing notebooks and job scripts.
