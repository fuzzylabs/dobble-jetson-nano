import os
import json
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
from typing import Optional, Sequence
from PIL import Image
import random

IMAGE_BASE_DIR="data/dobble/images"
VOC_FOLDER="data/dobble/voc"

@dataclass
class VOCObject:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

@dataclass
class Annotation:
    folder: str
    filename: str
    path: str
    width: int
    height: int
    objects: Sequence[VOCObject]

def get_labels(labelbox_export) -> Sequence[str]:
    labels = []
    for image_id, card in enumerate(labelbox_export):
        if "objects" in card["Label"]:
            for obj in card["Label"]["objects"]:
                labels += [obj["value"]]
    return list(set(labels))

def get_annotations(labelbox_export) -> Sequence[Annotation]:
    annotations = []
    for image_id, card in enumerate(labelbox_export):
        folder = IMAGE_BASE_DIR
        filename = card["External ID"]
        _image = Image.open(f"{IMAGE_BASE_DIR}/{filename}")
        width, height = (_image.width, _image.height)
        _image.save(f"{VOC_FOLDER}/JPEGImages/{filename}")
        if "objects" in card["Label"]:
            objects = []
            for obj in card["Label"]["objects"]:
                category_name = obj["value"]
                labelbox_bbox = obj["bbox"]
                xmin = labelbox_bbox["left"]
                ymin = labelbox_bbox["top"]
                xmax = xmin + labelbox_bbox["width"]
                ymax = ymin + labelbox_bbox["height"]

                objects += [
                    VOCObject(
                        name=category_name,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                    )
                ]

            annotations += [Annotation(
                folder = folder,
                filename = filename,
                path = f"{IMAGE_BASE_DIR}/{filename}",
                width = width,
                height = height,
                objects = objects,
            )]
        else:
            continue


    return annotations

def create_training_directories():
    os.makedirs(f"{VOC_FOLDER}/Annotations", exist_ok=True)
    os.makedirs(f"{VOC_FOLDER}/ImageSets/Main", exist_ok=True)
    os.makedirs(f"{VOC_FOLDER}/JPEGImages", exist_ok=True)

def voc_object_to_xml(obj: VOCObject):
    xml = ""
    xml += "<object>\n"
    xml += f"<name>{obj.name}</name>\n"
    xml += f"<pose>Unspecified</pose>\n"
    xml += f"<truncated>0</truncated>\n"
    xml += f"<difficult>0</difficult>\n"

    xml += "<bndbox>\n"
    xml += f"<xmin>{obj.xmin}</xmin>\n"
    xml += f"<ymin>{obj.ymin}</ymin>\n"
    xml += f"<xmax>{obj.xmax}</xmax>\n"
    xml += f"<ymax>{obj.ymax}</ymax>\n"

    xml += "</bndbox>\n"

    xml += "</object>\n"
    return xml

def save_annotations_to_xml(annotations: Sequence[Annotation]):
    for i, annotation in enumerate(annotations):
        xml = ""
        xml += "<annotation>\n"
        xml += f"<folder>{annotation.folder}</folder>\n"
        xml += f"<filename>{annotation.filename}</filename>\n"
        xml += f"<path>{annotation.path}</path>\n"
        xml += f"<source><database>Dobble</database></source>\n"

        xml += "<size>\n"
        xml += f"<width>{annotation.width}</width>\n"
        xml += f"<height>{annotation.height}</height>\n"
        xml += f"<depth>3</depth>\n"
        xml += "</size>\n"
        
        xml += "<segmented>0</segmented>\n"
        for obj in annotation.objects:
            xml += voc_object_to_xml(obj)
        
        xml += "</annotation>\n"
        with open(f"{VOC_FOLDER}/Annotations/{annotation.filename.split('.')[0]}.xml", "w") as f:
            f.write(xml)

def save_labels(labels: Sequence[str]):
    with open(f"{VOC_FOLDER}/labels.txt", "w") as f:
        for label in labels:
            f.write(f"{label}\n")

def save_default_image_set(annotations: Sequence[Annotation]):
    random.shuffle(annotations)
    cutoff = int(len(annotations) * 0.8)
    trainval = annotations[:cutoff]
    test = annotations[cutoff:]

    with open(f"{VOC_FOLDER}/ImageSets/Main/trainval.txt", "w") as f:
        for a in trainval:
            f.write(a.filename.split('.')[0] + "\n")

    with open(f"{VOC_FOLDER}/ImageSets/Main/test.txt", "w") as f:
        for a in test:
            f.write(a.filename.split('.')[0] + "\n")

with open("data/dobble/labelbox_export.json") as f:
    labelbox_export = json.load(f)

create_training_directories()

annotations = get_annotations(labelbox_export)
labels = get_labels(labelbox_export)

save_annotations_to_xml(annotations)
save_labels(labels)
save_default_image_set(annotations)