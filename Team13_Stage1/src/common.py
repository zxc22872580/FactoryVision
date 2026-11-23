"""Shared constants / data listing helpers for Stage 1.

No absolute personal paths: the dataset root is always passed in by the caller
(``--data_root``), and the checkpoint is resolved relative to this file.
"""
import os
import random

import numpy as np

CLASS_ORDER = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled_in_scale", "scratches"]
NUM_CLASSES = len(CLASS_ORDER)
SEED = 42
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Grayscale statistics of the FactoryVision classification training split,
# computed once over the 1,260 training images only (never over validation):
#     mean = sum(pixel) / N, std = sqrt(E[x^2] - E[x]^2), pixels scaled to [0,1]
# They are hard-coded so that training and benchmark inference always use the
# exact same normalisation.
NORM_MEAN = [0.5070]
NORM_STD = [0.2132]
INPUT_SIZE = 128


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
    """List (paths, labels) of a class-folder split, deterministically sorted."""
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
    """List image files of a flat folder (used by the benchmark inference)."""
    return [os.path.join(folder, fn) for fn in sorted(os.listdir(folder))
            if fn.lower().endswith(IMG_EXT)]
