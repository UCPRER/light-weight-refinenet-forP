#!/bin/sh
PYTHONPATH=$(pwd):$PYTHONPATH python src_v2/train.py \
    --enc-backbone 50 \
    --num-stages 1 \
    --num-classes 21 \
    --train-dir './datasets/' \
    --val-dir './datasets/' \
    --dataset-type 'torchvision' \
    --stage-names 'VOC' \
    --augmentations-type 'albumentations'


# Uncomment below to download datasets using torchvision API
# --train-download 1 1 \
# --val-download 1