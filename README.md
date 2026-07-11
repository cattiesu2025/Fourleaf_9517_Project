# COMP9517 Evaluation and Robustness MVP

This branch contains the first runnable version of the shared Evaluation and
Robustness framework due on 16 July.

## Included deliverables

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

## Environment setup

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

## Run the evaluation demo

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

## Run the dummy-predictor robustness demo

```bash
python demo/evaluation_mvp/dummy_robustness.py \
  --output tmp/mvp-dummy-robustness \
  --force
```

This command runs one dummy predictor through four degradation types and five
severity levels. It creates 20 standard result directories and an aggregated
robustness summary.

## Run the tests

```bash
python -m pytest
```

The test suite covers metric calculation, `image_id` alignment, permuted
`class_indices`, confusion-matrix outputs, deterministic degradation, the
shared predictor interface, and the complete 4x5 robustness matrix.
