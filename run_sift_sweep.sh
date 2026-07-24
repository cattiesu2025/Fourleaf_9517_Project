#!/bin/bash
# run_sift_sweep.sh
# Usage: bash run_sift_sweep.sh
set -e   # stop immediately if any command fails, instead of silently continuing

mkdir -p logs

for V in 500 1000 2000; do
  for C in 0.1 1 10; do
    echo "=== running vocab_size=$V svm_c=$C ==="
    python -m src.traditional.sift_bovw_svm \
        --train_csv data/metadata/train.csv \
        --test_csv  data/metadata/val.csv \
        --vocab_size "$V" \
        --max_desc_per_image 200 \
        --max_total_desc 1000000 \
        --svm_c "$C" \
        --cache_dir outputs/traditional/sift_bovw_svm/feature_cache \
        --output_dir "outputs/traditional/sift_bovw_svm_dev_v${V}_c${C}" \
        > "logs/sift_v${V}_c${C}.log" 2>&1
    echo "finished vocab=$V svm_c=$C"
  done
done

echo "all SIFT runs complete"
