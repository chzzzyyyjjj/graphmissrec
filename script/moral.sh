#!/bin/bash
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=$1 python run_missing_train_experiments.py \
  --dataset Pantry_mm_full \
  --pretrained saved/MISSRec-FHCKM_mm_full-100.pth \
  --ratios 0,0.1,0.2,0.3 \
  --missing-modes img \
  --gpu 1
