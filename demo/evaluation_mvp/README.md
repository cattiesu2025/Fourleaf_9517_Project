# Evaluation and robustness MVP demo

This package contains small sample inputs and numerical evaluation fixtures. No
real dataset or trained model is required.

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
|-- per_class_metrics.csv
|-- top_confused_pairs.csv
`-- failure_cases.csv
```

The `scores.npz` rows and `class_indices` are deliberately out of order. A successful run therefore also demonstrates `image_id` alignment and class-column mapping.

## Run the evaluation demo

From the project root:

```bash
./scripts/comp9517 evaluate \
  --predictions demo/evaluation_mvp/input/predictions.csv \
  --scores demo/evaluation_mvp/input/scores.npz \
  --runtime demo/evaluation_mvp/input/runtime.json \
  --classes demo/evaluation_mvp/input/idx_to_class.json \
  --output tmp/mvp-evaluation-output \
  --expected-num-classes 3
```

The generated values should match `expected_output/metrics.json` and `expected_output/confusion_matrix.npy`.
Generated plot images remain local and are ignored by Git.

## Run the dummy-predictor robustness demo

```bash
./scripts/comp9517 demo-robustness \
  --output tmp/mvp-dummy-robustness \
  --force
```

This uses the shared `ModelPredictor` interface to run a tiny colour-based model through all four degradations and five severity levels. It produces 20 standard inference directories plus `robustness_summary.csv` and `robustness_curves.png`.

## Regenerate the sample package

```bash
./scripts/comp9517 build-demo
```
