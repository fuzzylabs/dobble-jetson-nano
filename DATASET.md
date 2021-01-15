# Prepare dataset

The following is how we have prepared the dataset that is published to Kaggle.

TODO: Add Kaggle URL.

Our dataset is built on top of [https://www.kaggle.com/grouby/dobble-card-images](https://www.kaggle.com/grouby/dobble-card-images). We start with this dataset - label each symbol in [labelbox](), then export those labels and convert the files into the required format for training.

## Conda Environment

We use Conda in order to provide a consistent python build environment for preparing our dataset.

### Install Conda (aarch64)

```
wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh
bash Archiconda3-0.2.3-Linux-aarch64.sh
```

### Setup Conda environment

```
conda env create -f environment-jetson.yml
conda activate dobble
```

## Download source dataset

Pre-requisites: a Kaggle account is required to use the `kaggle` command line tool. Follow the instructions [here](https://www.kaggle.com/docs/api) to acquire a Kaggle API token and save this locally.

Alternatively, you can download the data directly from the Kaggle website and unzip it to the `data` directory.

```sh
kaggle datasets download -d grouby/dobble-card-images -p data
cd data
unzip dobble-card-images.zip
```

### Notes on the source Kaggle dataset

Things to bare in mind

* Text files are DOS-formatted, so have Windows/DOS style line endings.
* `dobble_card_symbol_mappings.txt` and `dobble_symbols.txt` begin with a byte-order mark `\ufeff`.

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
