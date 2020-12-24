"""
This script uses the Kaggle dataset to generate a image classification training dataset for PyTorch

* Each training example is an image with two Dobble cards
* The training examples are labelled according to which symbol matches
"""

from typing import Sequence, Set, Dict
from dataclasses import dataclass
import io

DATA_BASE_DIR = "data"

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
        for l in f:
            result.append(clean_string(l))
    return result

def label_cards(mappings: Sequence[str], labels: Dict[str, str]) -> Sequence[LabelledCard]:
    result = []
    for m in mappings:
        split_m = m.split(",")
        card_id = split_m[0]
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
            print(f"{left_card.labels}, {right_card.labels} -> {left_card.labels.intersection(right_card.labels)}")
            common_symbol = left_card.labels.intersection(right_card.labels)
            result.append(TrainingExample(left_card.card, right_card.card, common_symbol))

    return result

labels = read_labels()
print(f"Read {len(labels)} labels")

card_symbol_mappings = read_card_symbol_mappings()
print(f"Read {len(card_symbol_mappings)} card-symbol mappings")

labelled_cards = label_cards(card_symbol_mappings, labels)
print(f"Labelled all cards {labelled_cards[:3]}")

training_set = generate_training_set(labelled_cards)
print(training_set[0])
