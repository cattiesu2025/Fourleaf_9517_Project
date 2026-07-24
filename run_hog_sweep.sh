#!/bin/bash
# run_hog_sweep.sh
# Usage: bash run_hog_sweep.sh
#
# max_depth is capped (was previously unbounded) because on a 500-class
# problem, unbounded RandomForest trees can grow very large and exhaust
# memory, causing the process to be silently killed by the OS (no traceback,
# no error message -- the process just disappears). n_jobs is also capped
# instead of using all cores (-1), to reduce peak memory during parallel
# tree construction.
set -e

mkdir -p logs

for CELL in 16 8; do
  echo "=== running pixels_per_cell=$CELL ==="
  python -m src.traditional.hog_random_forest \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/val.csv \
      --pixels_per_cell "$CELL" \
      --n_estimators 200 \
      --max_depth 30 \
      --n_jobs 4 \
      --cache_dir outputs/traditional/hog_random_forest/feature_cache \
      --output_dir "outputs/traditional/hog_random_forest_dev_cell${CELL}" \
      > "logs/hog_cell${CELL}.log" 2>&1
  echo "finished pixels_per_cell=$CELL"
done

for N in 100 200 300; do
  echo "=== running n_estimators=$N (pixels_per_cell=8, max_depth=30) ==="
  python -m src.traditional.hog_random_forest \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/val.csv \
      --pixels_per_cell 8 \
      --n_estimators "$N" \
      --max_depth 30 \
      --n_jobs 4 \
      --cache_dir outputs/traditional/hog_random_forest/feature_cache \
      --output_dir "outputs/traditional/hog_random_forest_dev_n${N}" \
      > "logs/hog_n${N}.log" 2>&1
  echo "finished n_estimators=$N"
done

for D in 20 30 50; do
  echo "=== running max_depth=$D (pixels_per_cell=8, n_estimators=200) ==="
  python -m src.traditional.hog_random_forest \
      --train_csv data/metadata/train.csv \
      --test_csv  data/metadata/val.csv \
      --pixels_per_cell 8 \
      --n_estimators 200 \
      --max_depth "$D" \
      --n_jobs 4 \
      --cache_dir outputs/traditional/hog_random_forest/feature_cache \
      --output_dir "outputs/traditional/hog_random_forest_dev_d${D}" \
      > "logs/hog_d${D}.log" 2>&1
  echo "finished max_depth=$D"
done

echo "all HOG runs complete"