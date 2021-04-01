#!/usr/bin/env bash
#
# Start an instance of the dobble-jetson-nano docker container.
# See below or run this script with -h or --help to see usage options.
#
# This script should be run from the root dir of the dobble-jetson-nano project:
#
#     $ docker/run.sh
#

set -x

show_help() {
    echo " "
    echo "usage: Starts the Docker container and runs a user-specified command"
    echo " "
    echo "   ./docker/run.sh --container DOCKER_IMAGE"
    echo "                   --volume HOST_DIR:MOUNT_DIR"
    echo "                   --run RUN_COMMAND"
    echo " "
    echo "args:"
    echo " "
    echo "   --help                       Show this help text and quit"
    echo " "
    echo "   -v, --volume HOST_DIR:MOUNT_DIR Mount a path from the host system into"
    echo "                                   the container.  Should be specified as:"
    echo " "
    echo "                                      -v /my/host/path:/my/container/path"
    echo " "
    echo "                                   (these should be absolute paths)"
    echo " "
    echo "   -s, --shell  Run container with shell instead of the inference script"
    echo " "
    echo "   -X  Enable X/GUI forwarding"
    echo " "
    echo "   -c, --camera  Mount camera devices"
    echo " "
    echo "   -t, --train Run training instead of inference (./data/dobble and ./models directories are mounted. SSD base model will be downloaded to ./models/ssd if it is not present)"
    echo " "
}

die() {
    printf '%s\n' "$1"
    show_help
    exit 1
}

while :; do
    case $1 in
        -h|-\?|--help)
            show_help    # Display a usage synopsis.
            exit
            ;;
        -v|--volume)
            if [ "$2" ]; then
                USER_VOLUME=" -v $2 "
                shift
            else
                die 'ERROR: "--volume" requires a non-empty option argument.'
            fi
            ;;
        --volume=?*)
            USER_VOLUME=" -v ${1#*=} " # Delete everything up to "=" and assign the remainder.
            ;;
        --volume=)         # Handle the case of an empty --volume=
            die 'ERROR: "--volume" requires a non-empty option argument.'
            ;;
        -s|--shell)
            SHELL_ENTRYPOINT="--entrypoint /bin/bash"
            ;;
        -X)
            XORG_FORWARD="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v /tmp/argus_socket:/tmp/argus_socket"
            ;;
        -c|--camera)
            # check for V4L2 devices
            V4L2_DEVICES=" "

            for i in {0..9}
            do
                if [ -a "/dev/video$i" ]; then
                    V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i "
                fi
            done

            echo "V4L2_DEVICES:  $V4L2_DEVICES"
            ;;
        -t)
            TRAINING_ENTRYPOINT="--entrypoint ./train_object_detection.sh"
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac

    shift
done

CONTAINER_IMAGE="fuzzylabs/dobble-jetson-nano:latest"

DOBBLE_DOCKER_ROOT="/dobble-jetson-nano" # where this project lives

# If training SSD base model needs to be downloaded, and data and model directories need to be mounted
if [ -n "$TRAINING_ENTRYPOINT" ]; then
    # check for pytorch-ssd base model
    SSD_BASE_MODEL="$PWD/models/ssd/mobilenet-v1-ssd-mp-0_675.pth"

    if [ ! -f "$SSD_BASE_MODEL" ]; then
        echo "Downloading pytorch-ssd base model..."
        mkdir -p "$PWD/models/ssd/"
        wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O $SSD_BASE_MODEL
    fi

    # generate mount commands
    TRAINING_VOLUMES="\
    --volume $PWD/data/dobble:$DOBBLE_DOCKER_ROOT/data/dobble \
    --volume $PWD/models:$DOBBLE_DOCKER_ROOT/models"
fi

# set xhost for X forwarding
if [ -n "$XORG_FORWARD" ]; then
    sudo xhost +si:localuser:root
fi


sudo docker run --runtime nvidia -it --rm \
    $XORG_FORWARD \
    $TRAINING_ENTRYPOINT $TRAINING_VOLUMES \
    $SHELL_ENTRYPOINT \
    $V4L2_DEVICES \
    $CONTAINER_IMAGE
