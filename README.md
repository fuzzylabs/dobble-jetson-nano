# Training

## Setup

```sh
conda create -n dobble-jetson-nano python=3.8.5
conda activate dobble-jetson-nano
```

## Download dataset

```sh
kaggle datasets download -d grouby/dobble-card-images -p data
cd data
unzip dobble-card-images.zip
```
