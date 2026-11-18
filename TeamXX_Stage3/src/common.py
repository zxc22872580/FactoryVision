"""Shared constants / helpers for Stage 3 (Object Detection)."""
import os
import random
import xml.etree.ElementTree as ET

import numpy as np

CLASS_ORDER = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled_in_scale", "scratches"]
NUM_CLASSES = len(CLASS_ORDER)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
# The NEU-DET VOC annotations spell this class with a hyphen, while the
# benchmark output format requires the underscore form. Accept both on input
# and always emit the underscore form.
CLASS_ALIASES = {"rolled-in_scale": "rolled_in_scale",
                 "rolled_in_scale": "rolled_in_scale"}


SEED = 42
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def canonical_class(name):
    """Map an annotation class name onto the benchmark's spelling."""
    return CLASS_ALIASES.get(name.strip(), name.strip())


INPUT_SIZE = 640
CONF_THRESHOLD = 0.001    # low on purpose: AP integrates the full PR curve
                          # (measured on validation: 0.001 > 0.005 > 0.01 > 0.05)
NMS_IOU_THRESHOLD = 0.6
MAX_DET = 100             # hard format limit of the benchmark


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def list_images(folder):
    return [os.path.join(folder, fn) for fn in sorted(os.listdir(folder))
            if fn.lower().endswith(IMG_EXT)]


def parse_voc(xml_path):
    """Return (width, height, [(class_idx, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = canonical_class(obj.find("name").text)
        if name not in CLASS_TO_IDX:
            continue
        bb = obj.find("bndbox")
        boxes.append((CLASS_TO_IDX[name],
                      float(bb.find("xmin").text), float(bb.find("ymin").text),
                      float(bb.find("xmax").text), float(bb.find("ymax").text)))
    return w, h, boxes


def load_ground_truth(xml_dir):
    """image_id -> {'size': (w,h), 'boxes': [(cls, x1,y1,x2,y2), ...]}"""
    gt = {}
    for fn in sorted(os.listdir(xml_dir)):
        if not fn.lower().endswith(".xml"):
            continue
        w, h, boxes = parse_voc(os.path.join(xml_dir, fn))
        gt[os.path.splitext(fn)[0]] = {"size": (w, h), "boxes": boxes}
    return gt
