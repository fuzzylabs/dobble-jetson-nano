# Introduction

Dobble is a game with the aim to recognise matching symbols in a pair of disc-shaped cards. In this repository we aim to train an image classifier that can identify the common symbol in a pair of cards.

Inspiration from https://www.hackster.io/aidventure/the-dobble-challenge-93d57c

## Inference
# Conda Environment

The aarch64 version of conda can be got from [https://github.com/Archiconda/build-tools/releases](https://github.com/Archiconda/build-tools/releases)

Conda environment then can be set up 

```
conda env create -f environment-jetson.yml
conda activate dobble
```

# Custom Docker container
To build:
```
docker/build.sh
```

To start:
```
docker/run.sh
```

# Object detection

## Image preparation and labeling
Images need to be converted from TIF to JPEG for labeling (ImageMagick's `convert` tool is required)
```
./convert-images.sh
```

The resulting images are saved at `data/dobble/images/deck*_card*.jpg`. They now can be imported to LabelBox. 

After the labeling is done in LabelBox, export the results to `data/dobble/labelbox_export.json`

## Dataset preparation
In dobble-jetson-nano directory

Required: 
* `data/dobble/labelbox_export.json` -- a labeled set from Labelbox
* `data/dobble/images/*.jpg` -- images used for labeling

```
python3 labelbox_to_voc.py
```

Saves VOC format dataset to `data/dobble/voc`.

Run with the provided model or train it yourself, see training section below.

```
docker/run.sh
# Starts Docker container with shell
python detect-dobble.py $SOURCE
```
$SOURCE can be an image, a video, or a video device.

## Training

### Obtain the dataset from Kaggle

TODO: Add instructions to download our dataset

If you're interested in how we prepared the dataset see [DATASET.md](DATASET.md).

### Run the training

```
docker/run.sh
# Starts Docker container with shell
./train_object_detection.sh
```
