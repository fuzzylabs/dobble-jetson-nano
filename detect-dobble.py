import jetson.inference
import jetson.utils

NET_DIR="models/dobble"

print(f"--model {NET_DIR}/ssd-mobilenet.onnx --labels {NET_DIR}/labels.txt")

net = jetson.inference.detectNet(
    argv=[f"--model={NET_DIR}/ssd-mobilenet.onnx", f"--labels={NET_DIR}/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"],
    threshold=0.5
)

_input = jetson.utils.videoSource("/dobble-jetson-nano/data/dobble/voc/JPEGImages/card50_01.jpg")
img = _input.Capture()

detections = net.Detect(img)

def are_overlapping(a, b):
    ax0, ax1, ay0, ay1 = a.Left, a.Right, a.Top, a.Bottom
    bx0, bx1, by0, by1 = b.Left, b.Right, b.Top, b.Bottom

    return not (bx0 > ax1 or ax0 > bx1 or by0 > ay1 or ay0 > by1)

def remove_overlaps(detections):
    _detections = []
    for i, a in enumerate(detections):
        overlaps = False
        for b in detections[i+1:]:
            if a.ClassID == b.ClassID and are_overlapping(a, b):
                overlaps = True
                break
        if not overlaps:
            _detections += [a]

    return _detections

print([net.GetClassDesc(x.ClassID) for x in remove_overlaps(detections)])
