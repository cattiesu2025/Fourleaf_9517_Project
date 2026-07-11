# Evaluation and robustness MVP demo

This package contains ready-to-open sample inputs and their expected evaluation outputs. No real dataset or trained model is required.

## Included files

```text
input/
|-- predictions.csv
|-- scores.npz
|-- idx_to_class.json
`-- runtime.json

expected_output/
|-- metrics.json
|-- confusion_matrix.npy
|-- confusion_matrix.png
|-- confusion_matrix_top_classes.png
|-- per_class_metrics.csv
|-- top_confused_pairs.csv
`-- failure_cases.csv
```

The `scores.npz` rows and `class_indices` are deliberately out of order. A successful run therefore also demonstrates `image_id` alignment and class-column mapping.

## Run the evaluation demo

From the project root:

```bash
python -m src.evaluation.evaluate \
  --predictions demo/evaluation_mvp/input/predictions.csv \
  --scores demo/evaluation_mvp/input/scores.npz \
  --runtime demo/evaluation_mvp/input/runtime.json \
  --classes demo/evaluation_mvp/input/idx_to_class.json \
  --output tmp/mvp-evaluation-output \
  --expected-num-classes 3
```

The generated values should match `expected_output/metrics.json` and `expected_output/confusion_matrix.npy`.

## Run the dummy-predictor robustness demo

```bash
python demo/evaluation_mvp/dummy_robustness.py \
  --output tmp/mvp-dummy-robustness \
  --force
```

This uses the shared `ModelPredictor` interface to run a tiny colour-based model through all four degradations and five severity levels. It produces 20 standard inference directories plus `robustness_summary.csv` and `robustness_curves.png`.

## Regenerate the checked-in sample package

```bash
python scripts/build_mvp_demo.py
```
