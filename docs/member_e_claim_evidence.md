# Member E claim--evidence ledger

This ledger records the evidence behind Member E's quantitative claims.  Raw
Drive artifacts remain outside Git; the generated CSV/JSON files are the
reproducible, repository-sized summaries.

## Evaluation invariants

- **Shared test set:** all 12 registered runs contain 5,000 unique test images
  and 500 classes.  The evaluator verifies an identical `image_id ->
  true_label` mapping before producing tables.
- **Evidence:** `report/generated/evaluation_evidence.json` and the validation
  logic in `src/evaluation/compare.py`.
- **Reproduction:** run `python -m src.evaluation.compare --manifest
  configs/final_evaluation.yaml --artifact-root <drive-snapshot>
  --output report/generated`.

## Limited-data main comparison

| Claim | Evidence |
| --- | --- |
| SIFT + BoVW + SVM reaches 2.94% Top-1, 8.02% Top-5, and 2.26% Macro-F1. | `report/generated/main.csv`, run `sift_bovw_svm` |
| HOG + RF reaches 1.38% Top-1, 5.26% Top-5, and 0.89% Macro-F1. | `report/generated/main.csv`, run `hog_random_forest` |
| Best scratch reaches 37.94% Top-1 and 37.51% Macro-F1. | `report/generated/main.csv`, run `scratch_basic_aug_sgd` |
| Frozen transfer reaches 53.88% Top-1; its gain over scratch is 15.94 pp. | `report/generated/main.csv`; subtraction `53.88 - 37.94` |
| Fine-tuning reaches 66.24% Top-1 and 66.20% Macro-F1. | `report/generated/main.csv`, run `transfer_finetuned` |
| Fine-tuning gains 12.36 pp over frozen and 28.30 pp over scratch in Top-1. | `66.24 - 53.88`; `66.24 - 37.94` |
| Limited-data runtimes are not a controlled hardware comparison. | Per-run `runtime.json`; hardware field differs across B/C/D artifacts. |

## Scratch and transfer ablations

| Claim | Evidence |
| --- | --- |
| Basic augmentation improves Top-1 from 22.62% to 31.76% (+9.14 pp). | `report/generated/scratch_ablation.csv` |
| Strong augmentation reaches 25.88%, below basic augmentation. | `report/generated/scratch_ablation.csv` |
| Regularized transfer reaches 65.20% Top-1 and 65.11% Macro-F1. | `report/generated/transfer_ablation.csv` |
| Self-attention reaches 62.08% Top-1 and 62.04% Macro-F1. | `report/generated/transfer_ablation.csv` |

## Full-data extension

| Claim | Evidence |
| --- | --- |
| Full-data scratch reaches 64.20% Top-1, 84.76% Top-5, and 64.02% Macro-F1. | `report/generated/full_data.csv`, run `full_scratch` |
| Full-data fine-tuning reaches 81.58% Top-1, 93.70% Top-5, and 81.53% Macro-F1. | `report/generated/full_data.csv`, run `full_finetuned` |
| Fine-tuning gains 17.38 pp Top-1, 8.94 pp Top-5, and 17.52 pp Macro-F1. | Pairwise subtraction of the two rows in `report/generated/full_data.csv` |
| Training takes 1,295.0 s vs 1,351.7 s: +56.7 s or approximately +4.38%. | `report/generated/full_data.csv`; `(1351.68 - 1295.00) / 1295.00` |
| Best validation epochs are 15 for scratch and 13 for fine-tuning. | Source `training_history.csv` files listed in `configs/final_evaluation.yaml` |
| The comparison is matched by GPU model, not physical node. | Source runtime metadata: L40S nodes `k098` and `k097`. |

## Robustness

The final 80-row matrix comes from `scripts/run_final_robustness.py` using the
original RGB test images, four degradation families, five severities, and the
fixed models in `configs/final_robustness.yaml`.  The tracked summary is
`report/generated/robustness_summary.csv`; the paper figure is
`report/generated/robustness_curves.pdf`.

| Claim | Evidence |
| --- | --- |
| Fine-tuned ResNet18 has the highest Top-1 in all 20 degraded conditions. | Group-wise maximum over `report/generated/robustness_summary.csv` |
| At maximum blur, Fine-tuned falls 66.24% -> 9.02% and Scratch 37.94% -> 3.90%. | `degradation_type=blur`, `severity=5` |
| At brightness factor 0.5, Fine-tuned obtains 59.92% and Scratch 16.70%. | `degradation_type=brightness`, `severity=5` |
| At noise sigma 0.10, Fine-tuned obtains 44.24% and Scratch 27.70%. | `degradation_type=gaussian_noise`, `severity=5` |
| At JPEG quality 10, Fine-tuned obtains 50.12% and Scratch 27.98%. | `degradation_type=jpeg_compression`, `severity=5` |
| Maximum-severity retention (blur/brightness/noise/JPEG) is 13.6/90.5/66.8/75.7% for Fine-tuned and 10.3/44.0/73.0/73.7% for Scratch. | Severity-5 Top-1 divided by each local clean Top-1 |
| HOG is most affected by noise (1.32% -> 0.60%); SIFT by blur (3.18% -> 1.26%). | Local severity-0 and severity-5 rows |

The original 5,000 test images are now available locally and all four selected
model adapters have completed a same-environment clean pass.  The two ResNet18
adapters reproduce all 5,000 submitted predictions.  HOG and SIFT show small
Top-1 drift on ARM macOS (1.32% vs 1.38%, and 3.18% vs 2.94%) despite matching
the recorded feature-extraction library versions; therefore degradation drops use each model's
local severity-0 output rather than mixing local degraded inference with
submitted x86 clean scores.  These preflight outputs are stored locally under
`outputs/member_e_robustness/*/clean/severity_0/` and remain gitignored.
