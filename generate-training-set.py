"""
This script uses the Kaggle dataset to generate a image classification training dataset for PyTorch

* Each training example is an image with two Dobble cards
* The training examples are labelled according to which symbol matches
"""

from typing import Sequence, Set, Dict
from dataclasses import dataclass
import io
import os
from PIL import Image
import random
from math import floor
from tqdm import tqdm

DATA_BASE_DIR = "data"
TRAIN_BASE_DIR = "train"
TRAINING_IMAGE_DIMENSIONS = (600, 300)


@dataclass
class LabelledCard:
    card: str
    labels: Set[str]


@dataclass
class TrainingExample:
    left_card: str
    right_card: str
    label: str


def clean_string(s: str) -> str:
    return s.replace("\n", "").replace("\ufeff", "")


def read_labels() -> Dict[str, str]:
    result = {}
    with io.open(f"{DATA_BASE_DIR}/dobble_symbols.txt") as f:
        for line in f:
            key, val = clean_string(line).split(",")
            result[key] = val
    return result


def read_card_symbol_mappings() -> Sequence[str]:
    result = []
    with io.open(f"{DATA_BASE_DIR}/dobble_card_symbol_mapping.txt") as f:
        for line in f:
            result.append(clean_string(line))
    return result


def label_cards(mappings: Sequence[str], labels: Dict[str, str]) -> Sequence[LabelledCard]:
    result = []
    for m in mappings:
        split_m = m.split(",")
        card_id = split_m[0].zfill(2)
        card_labels = set()
        for idx, v in enumerate(split_m[1:]):
            if v == "1":
                card_labels.add(labels[str(idx + 1)])
        result.append(LabelledCard(card_id, card_labels))

    return result


def generate_training_set(cards: Sequence[LabelledCard]) -> Sequence[TrainingExample]:
    result = []
    for left_card in cards:
        for right_card in cards:
            if left_card != right_card:
                common_symbol = left_card.labels.intersection(
                    right_card.labels).pop()
                result.append(TrainingExample(left_card.card,
                                              right_card.card, common_symbol))

    return result


def generate_training_image(training_example: TrainingExample, deck: str) -> Image:
    left_n = training_example.left_card
    left_im = Image.open(
        f"{DATA_BASE_DIR}/{deck}/{left_n}/card{left_n}_01.tif")
    right_n = training_example.right_card
    right_im = Image.open(
        f"{DATA_BASE_DIR}/{deck}/{right_n}/card{right_n}_01.tif")

    left_im = left_im.resize(
        (TRAINING_IMAGE_DIMENSIONS[0] // 2, TRAINING_IMAGE_DIMENSIONS[1]))
    right_im = right_im.resize(
        (TRAINING_IMAGE_DIMENSIONS[0] // 2, TRAINING_IMAGE_DIMENSIONS[1]))

    result = Image.new("RGB", TRAINING_IMAGE_DIMENSIONS)
    result.paste(left_im, (0, 0))
    result.paste(right_im, (TRAINING_IMAGE_DIMENSIONS[0] // 2, 0))

    return result


def create_training_label_file(labels: Sequence[str]):
    with io.open(f"{TRAIN_BASE_DIR}/labels.txt", "w") as f:
        for label in labels:
            f.write(f"{label}\n")


def create_training_directories(labels: Sequence[str]):
    for label in labels:
        os.makedirs(f"{TRAIN_BASE_DIR}/train/{label}", exist_ok=True)
        os.makedirs(f"{TRAIN_BASE_DIR}/test/{label}", exist_ok=True)
        os.makedirs(f"{TRAIN_BASE_DIR}/val/{label}", exist_ok=True)

def group_training_examples_by_label(training_set: Sequence[TrainingExample]) -> Dict[str, TrainingExample]:
    training_example_groups = dict()
    for example in training_set:
        if example.label in training_example_groups:
            training_example_groups[example.label] += [example]
        else:
            training_example_groups[example.label] = [example]

    return training_example_groups

def save_training_example(example: TrainingExample, bucket: str):
    training_image = generate_training_image(
        example, "dobble_deck01_cards_57")
    label = example.label
    filename = f"{example.left_card}_{example.right_card}.png"
    training_image.save(
        f"{TRAIN_BASE_DIR}/{bucket}/{label}/{filename}.png")

def create_training_files(training_set: Sequence[TrainingExample], ):
    groups = group_training_examples_by_label(training_set)
    for group in groups:
        examples = groups[group]
        random.shuffle(examples)

        n = len(examples)
        train_split = floor(n * 0.6)
        test_split = floor(n * 0.8)

        for example in tqdm(examples[:train_split], desc=f"{group}_train"):
            save_training_example(example, bucket="train")

        for example in tqdm(examples[train_split:test_split], desc=f"{group}_test"):
            save_training_example(example, bucket="test")

        for example in tqdm(examples[test_split:], desc=f"{group}_val"):
            save_training_example(example, bucket="val")


labels = read_labels()
print(f"Read {len(labels)} labels")

card_symbol_mappings = read_card_symbol_mappings()
print(f"Read {len(card_symbol_mappings)} card-symbol mappings")

labelled_cards = label_cards(card_symbol_mappings, labels)
print(f"Labelled all cards")

training_set = generate_training_set(labelled_cards)
print(f"Training set has {len(training_set)} examples - {training_set[0]}")

#im = generate_training_image(training_set[0], "dobble_deck01_cards_57")
# im.show()

# Select a subset of labels
labels = dict(list(labels.items())[10:12])
training_set = [x for x in training_set if x.label in labels.values()]

create_training_directories(labels.values())
create_training_label_file(labels.values())
create_training_files(training_set)
