import os
from dataclasses import dataclass, field
from xml.dom import minidom
import jetson.inference
import jetson.utils
import pandas as pd
import numpy as np

VOC_FOLDER="data/dobble/voc"

@dataclass
class Box:
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def get_boxes_from_voc(name: str) -> Box:
    testfilename = os.path.join(VOC_FOLDER, f"Annotations/{name}.xml")
    doc = minidom.parse(testfilename)
    objects = doc.getElementsByTagName('object')
    boxes = []
    for obj in objects:
        label = obj.getElementsByTagName("name")[0].firstChild.nodeValue
        bndbox = obj.getElementsByTagName("bndbox")[0]
        xmin = bndbox.getElementsByTagName("xmin")[0].firstChild.nodeValue
        ymin = bndbox.getElementsByTagName("ymin")[0].firstChild.nodeValue
        xmax = bndbox.getElementsByTagName("xmax")[0].firstChild.nodeValue
        ymax = bndbox.getElementsByTagName("ymax")[0].firstChild.nodeValue
        boxes += [Box(label, float(xmin), float(ymin), float(xmax), float(ymax))]
    return boxes

def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = groundtruth_box.ymin, groundtruth_box.xmin, groundtruth_box.ymax, groundtruth_box.xmax
    d_ymin, d_xmin, d_ymax, d_xmax = detection_box.ymin, detection_box.xmin, detection_box.ymax, detection_box.xmax
    
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

# Read test set
with open(os.path.join(VOC_FOLDER, "ImageSets/Main/test.txt")) as f:
    test_set = [x.strip() for x in list(f)]

ground_truth = {}

# Read ground truth
for case in test_set:
    ground_truth[case] = get_boxes_from_voc(case)

# Perform object detection
NET_DIR="models/dobble"

net = jetson.inference.detectNet(
    argv=[f"--model={NET_DIR}/ssd-mobilenet.onnx", f"--labels={NET_DIR}/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"],
    threshold=0.5
)

def get_boxes_from_detections(detections):
    boxes = []
    for x in detections:
        label = net.GetClassDesc(x.ClassID)
        boxes += [Box(
            label,
            x.Left,
            x.Top,
            x.Right,
            x.Bottom,
        )]
    return boxes

predicted = {}

for case in test_set:
    image_filename = os.path.join(VOC_FOLDER, f"JPEGImages/{case}.jpg")
    _input = jetson.utils.videoSource(image_filename)
    img = _input.Capture()
    detections = net.Detect(img)
    predicted[case] = get_boxes_from_detections(detections)

pairs = [] # Pairs of (true, predicted) labels
iou_cutoff = 0.5
for case in test_set:
    pred = predicted[case]
    true = ground_truth[case]
    true_used = [False] * len(true)
    for pred_box in pred:
        found = False
        for true_i, true_box in enumerate(true):
            if compute_iou(true_box, pred_box) >= iou_cutoff:
                pairs += [(true_box.label, pred_box.label)]
                found = True
                true_used[true_i] = True
                break
        if not found:
            pairs += [("<nothing>", pred_box.label)]

    for used, true_box in zip(true_used, true):
        if not used:
            pairs += [(true_box.label, "<nothing>")]

print(pairs)

with open(f"{NET_DIR}/labels.txt") as f:
    labels = [x.strip() for x in list(f)[1:] + ["<nothing>"]]

print(labels)

confusion_matrix = pd.DataFrame(
    np.zeros((len(labels), len(labels))),
    columns=pd.Index(labels, name="Predicted"),
    index=pd.Index(labels, name="True"),
    dtype=int
)

for true, pred in pairs:
    confusion_matrix.loc[true, pred] += 1

print(confusion_matrix)
confusion_matrix.to_csv(os.path.join(NET_DIR, "confusion_matrix.csv"))