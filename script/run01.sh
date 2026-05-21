#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=$1 python finetune.py \
    -d Scientific_mm_subset \
    -mode transductive
cd -