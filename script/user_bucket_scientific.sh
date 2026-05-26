#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=$1 python user_bucket_eval.py \
    -d Scientific_mm_full \
    -p "${2:-saved/MISSRec_scientific.pth}" \
    --mode transductive \
    --separate-buckets \
    --eval-batch-size "${3:-256}" \
    --output-dir bucket_results \
    --show-progress
cd -
