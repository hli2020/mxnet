#!/usr/bin/env bash

gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

python train_end2end.py --dataset_path data_coco --network vgg --dataset coco --image_set train2014 --gpu $1
python test.py --dataset_path data_coco --network vgg --dataset coco --image_set val2014 --gpu $gpu
