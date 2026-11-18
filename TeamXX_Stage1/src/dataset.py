"""Stage 1 - dataset + transform definitions.

Baseline preprocessing is restricted to Resize / ToTensor / Normalize, as
required by the Stage 1 rules.  Improvement Method A (Data Augmentation) adds a
single, clearly documented augmentation pipeline on top of it.
"""
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common import INPUT_SIZE, NORM_MEAN, NORM_STD, list_split

# --- Improvement Method A : Data Augmentation hyper-parameters -------------
AUG_CONFIG = {
    "random_horizontal_flip_p": 0.5,
    "random_vertical_flip_p": 0.5,
    "random_rotation_degrees": 15,
    "random_resized_crop_scale": (0.75, 1.0),
    "random_resized_crop_ratio": (0.9, 1.111),
}


class SteelDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")
        return self.transform(img), self.labels[i]


def eval_transform(size=INPUT_SIZE):
    """Basic preprocessing only - also used at benchmark inference time."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def train_transform(augment=False, size=INPUT_SIZE):
    if not augment:
        return eval_transform(size)
    c = AUG_CONFIG
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size,
                                     scale=c["random_resized_crop_scale"],
                                     ratio=c["random_resized_crop_ratio"]),
        transforms.RandomHorizontalFlip(c["random_horizontal_flip_p"]),
        transforms.RandomVerticalFlip(c["random_vertical_flip_p"]),
        transforms.RandomRotation(c["random_rotation_degrees"]),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def build_datasets(data_root, augment=False, size=INPUT_SIZE,
                   merge_train_val=False):
    import os
    tr_paths, ytr = list_split(os.path.join(data_root, "classification",
                                            "train"))
    va_paths, yva = list_split(os.path.join(data_root, "classification",
                                            "validation"))
    if merge_train_val:
        tr_paths, ytr = tr_paths + va_paths, ytr + yva
    train_ds = SteelDataset(tr_paths, ytr, train_transform(augment, size))
    val_ds = SteelDataset(va_paths, yva, eval_transform(size))
    return train_ds, val_ds
