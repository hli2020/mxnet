#!/usr/bin/env bash

# run this experiment with
# nohup bash script/resnet_coco.sh 0,1 &> resnet_coco.log &
# to use gpu 0,1 to train, gpu 0 to test and write logs to resnet_coco.log
gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

python train_end2end.py --dataset_path data_coco --network resnet --dataset coco --image_set train2014 --gpu $1
python test.py --dataset_path data_coco --network resnet --dataset coco --image_set val2014 --gpu $gpu
