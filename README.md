# Training

Dobble is a game with the aim to recognise matching symbols in a pair of disc-shaped cards. In this repository we aim to train an image classifier that can perform the same task, identifying which symbol is common.

As an example, consider the following two images:

![](doc/example_1.png)
![](doc/example_2.png)

We'll first download a [dataset from Kaggle](https://www.kaggle.com/grouby/dobble-card-images). This dataset contains images of each Dobble card individually.

To turn this into a training set, we must generate pairings of cards and label them according to the common symbol. So for the above two images, we'll create a new image that includes both of those plus a label: _"stop sign"_.

## Setup

```sh
conda create -n dobble-jetson-nano python=3.8.5
conda activate dobble-jetson-nano
```

## Download dataset

Pre-requisites: a Kaggle account is required to use the `kaggle` command line tool. Follow the instructions [here](https://www.kaggle.com/docs/api) to acquire a Kaggle API token and save this locally.

Alternatively, you can download the data directly from the Kaggle website and unzip it to the `data` directory.

```sh
kaggle datasets download -d grouby/dobble-card-images -p data
cd data
unzip dobble-card-images.zip
```

## Generate training data

```sh
python generate-training-set.py
```

# Notes on the Kaggle dataset

Things to bare in mind

* Text files are DOS-formatted, so have Windows/DOS style line endings.
* `dobble_card_symbol_mappings.txt` and `dobble_symbols.txt` begin with a byte-order mark `\ufeff`.
