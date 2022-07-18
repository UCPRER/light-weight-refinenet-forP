#!/bin/sh
PYTHONPATH=$(pwd):$PYTHONPATH python src_v2/train.py \
    --enc-backbone 50 \
    --train-dir '/' \
    --val-dir '/' \
    --train-list-path '/home/ucprer/python/light-weight-refinenet/work_root/coco.train' \
    --val-list-path '/home/ucprer/python/light-weight-refinenet/work_root/coco.val' \
    --num-stages 1 \
    --num-classes 93 \
    --ignore-label 0 \
    --train-batch-size 8