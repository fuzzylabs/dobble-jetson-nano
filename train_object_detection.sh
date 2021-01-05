#!/usr/bin/env bash

SSD_BASE_PATH="/jetson-inference/python/training/detection/ssd"
TRAINING_SCRIPT="$SSD_BASE_PATH/train_ssd.py"
ONNX_EXPORT="$SSD_BASE_PATH/onnx_export.py"
PRETRAINED_SSD="$SSD_BASE_PATH/models/mobilenet-v1-ssd-mp-0_675.pth"
MODEL_DIR="/dobble-jetson-nano/models/dobble"
DATA_DIR="/dobble-jetson-nano/data/dobble/voc"

python3 $TRAINING_SCRIPT --pretrained-ssd $PRETRAINED_SSD --model-dir $MODEL_DIR --dataset-type voc --datasets $DATA_DIR --batch-size 2 --num-workers 1
python3 $ONNX_EXPORT --model-dir=$MODEL_DIR