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
Before building or running it is advised to have nvidia docker runtime to be set as the default:
```
# /etc/docker/daemon.json 
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```

## Build
```
./docker/build.sh
```
or
```
docker build -t fuzzylabsai/dobble-jetson-nano .
```

## Run
Use the provided script:
```
./docker/run.sh
```
which accepts multiple command line flags:
* `-v|--volume` -- similar to docker's `-v` flag, is used to mount volumes to the container
* `-s|--shell` -- change entrypoint to bash shell
* `-X` -- turn on X/GUI forwarding
* `-c|--camera` -- pass camera devices to the container
* `-t` -- change entrypoint to training instead of inference

Alternatively, you can execute `docker run` manually, providing the appropriate flags to docker.

```
docker run --runtime nvidia -it --rm $additional_flags fuzzylabsai/dobble-jetson-nano $source [$output]
```

The default entrypoint is the inference script, that requires a source (an image file, a video file or a camera device). The output is optional

Additional flags:
* `--device $DEV` e.g. `--device /dev/video0` -- to mount camera devices used as a source
* `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v /tmp/argus_socket:/tmp/argus_socket` -- pass DISPLAY environment variables and Xorg related files for GUI forwarding. 
* `-v $HOST_FILES:$CONTAINER_FILES` -- to mount files/directories, such as a directory with source files, and a directory to persist outputs to

An example of a full inference command:
```
docker run --runtime nvidia -it --rm --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v /tmp/argus_socket:/tmp/argus_socket -v `pwd`/examples:./examples fuzzylabsai/dobble-jetson-nano ./examples/source.mp4 ./example/
```

To start a shell in the container
```
docker run --runtime nvidia -it --rm $additional_flags --entrypoint /bin/bash fuzzylabsai/dobble-jetson-nano
```

To run training instead of inference
```
docker run --runtime nvidia -it --rm $additional_flags --entrypoint ./train_object_detection.sh fuzzylabsai/dobble-jetson-nano
```

## Overriding model
To override the model built into the Docker image, mount another model to `/docker-jetson-nano/models/dobble`
```
docker run ... -v $PATH_TO_MODEL:/docker-jetson-nano/models/dobble ...
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
docker/run.sh -t
```
