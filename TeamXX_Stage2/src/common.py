"""Shared constants / helpers for Stage 2 (Transfer Learning Challenge)."""
import os
import random

import numpy as np

CLASS_ORDER = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled_in_scale", "scratches"]
NUM_CLASSES = len(CLASS_ORDER)
SEED = 42
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ImageNet statistics - the pretrained backbones expect them.  The grayscale
# steel images are replicated to 3 channels before normalization.
INPUT_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def list_split(split_dir):
    paths, labels = [], []
    for idx, cname in enumerate(CLASS_ORDER):
        cdir = os.path.join(split_dir, cname)
        if not os.path.isdir(cdir):
            raise FileNotFoundError(cdir)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(IMG_EXT):
                paths.append(os.path.join(cdir, fn))
                labels.append(idx)
    return paths, labels


def list_images(folder):
    return [os.path.join(folder, fn) for fn in sorted(os.listdir(folder))
            if fn.lower().endswith(IMG_EXT)]
