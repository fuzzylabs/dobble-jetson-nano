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

## Generate training data

```sh
python generate-training-set.py
```

# Notes on the Kaggle dataset

Things to bare in mind

* Text files are DOS-formatted, so have Windows/DOS style line endings.
* `dobble_card_symbol_mappings.txt` and `dobble_symbols.txt` begin with a byte-order mark `\ufeff`.
