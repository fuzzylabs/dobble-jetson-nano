# Introduction

Dobble is a game with the aim to recognise matching symbols in a pair of disc-shaped cards. In this repository we aim to train an image classifier that can identify the common symbol in a pair of cards.

Inspiration from https://www.hackster.io/aidventure/the-dobble-challenge-93d57c

## Inference

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
