"""Stage 2 - dataset and the (single, shared) augmentation pipeline.

The *same* training augmentation and the *same* evaluation preprocessing are
used by all four 2x2 experiments, so the only things that differ between runs
are the architecture and the fine-tuning strategy.

The augmentation pipeline is deliberately built around the three robustness
benchmarks announced for Stage 2:

    lighting            -> ColorJitter(brightness/contrast) + RandomGamma
    optical             -> GaussianBlur / slight down-up resampling
    sensor_compression  -> JPEG re-compression + Gaussian sensor noise

They are applied at *training time only*; inference always uses the plain
Resize -> ToTensor -> Normalize path, identically for every benchmark folder.
"""
import io
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common import INPUT_SIZE, NORM_MEAN, NORM_STD, list_split

AUG_CONFIG = {
    "random_resized_crop_scale": (0.7, 1.0),
    "random_horizontal_flip_p": 0.5,
    "random_vertical_flip_p": 0.5,
    "random_rotation_degrees": 15,
    "color_jitter": {"brightness": 0.4, "contrast": 0.4, "p": 0.8},
    "random_gamma": {"range": (0.6, 1.6), "p": 0.3},
    "gaussian_blur": {"kernel": 5, "sigma": (0.3, 2.0), "p": 0.3},
    "downscale_upscale": {"factor": (0.4, 0.9), "p": 0.2},
    "jpeg_compression": {"quality": (25, 85), "p": 0.3},
    "gaussian_noise": {"std": (0.01, 0.07), "p": 0.3},
    "random_erasing": {"p": 0.25, "scale": (0.02, 0.15)},
}


class RandomGamma:
    def __init__(self, rng=(0.6, 1.6), p=0.3):
        self.rng, self.p = rng, p

    def __call__(self, img):
        if random.random() < self.p:
            g = random.uniform(*self.rng)
            img = transforms.functional.adjust_gamma(img, g)
        return img


class RandomDownUpScale:
    """Simulates an out-of-focus / low-resolution optical path."""

    def __init__(self, factor=(0.4, 0.9), p=0.2):
        self.factor, self.p = factor, p

    def __call__(self, img):
        if random.random() < self.p:
            f = random.uniform(*self.factor)
            w, h = img.size
            small = img.resize((max(8, int(w * f)), max(8, int(h * f))),
                               Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)
        return img


class RandomJPEG:
    """Simulates sensor / transmission compression artifacts."""

    def __init__(self, quality=(25, 85), p=0.3):
        self.quality, self.p = quality, p

    def __call__(self, img):
        if random.random() < self.p:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG",
                                    quality=random.randint(*self.quality))
            buf.seek(0)
            img = Image.open(buf).convert(img.mode)
        return img


class RandomGaussianNoise:
    """Applied on the tensor, before normalization."""

    def __init__(self, std=(0.01, 0.07), p=0.3):
        self.std, self.p = std, p

    def __call__(self, t):
        if random.random() < self.p:
            t = (t + torch.randn_like(t) * random.uniform(*self.std)).clamp(0, 1)
        return t


class SteelDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # grayscale source -> 3 identical channels for the ImageNet backbones
        img = Image.open(self.paths[i]).convert("L").convert("RGB")
        return self.transform(img), self.labels[i]


def eval_transform(size=INPUT_SIZE):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def train_transform(size=INPUT_SIZE, robust=True):
    c = AUG_CONFIG
    geo = [
        transforms.RandomResizedCrop(
            size, scale=c["random_resized_crop_scale"], ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(c["random_horizontal_flip_p"]),
        transforms.RandomVerticalFlip(c["random_vertical_flip_p"]),
        transforms.RandomRotation(c["random_rotation_degrees"]),
    ]
    if not robust:
        return transforms.Compose(
            geo + [transforms.ToTensor(),
                   transforms.Normalize(NORM_MEAN, NORM_STD)])
    photo = [
        transforms.RandomApply([transforms.ColorJitter(
            brightness=c["color_jitter"]["brightness"],
            contrast=c["color_jitter"]["contrast"])],
            p=c["color_jitter"]["p"]),
        RandomGamma(c["random_gamma"]["range"], c["random_gamma"]["p"]),
        transforms.RandomApply([transforms.GaussianBlur(
            c["gaussian_blur"]["kernel"], c["gaussian_blur"]["sigma"])],
            p=c["gaussian_blur"]["p"]),
        RandomDownUpScale(c["downscale_upscale"]["factor"],
                          c["downscale_upscale"]["p"]),
        RandomJPEG(c["jpeg_compression"]["quality"],
                   c["jpeg_compression"]["p"]),
    ]
    tail = [
        transforms.ToTensor(),
        RandomGaussianNoise(c["gaussian_noise"]["std"],
                            c["gaussian_noise"]["p"]),
        transforms.Normalize(NORM_MEAN, NORM_STD),
        transforms.RandomErasing(p=c["random_erasing"]["p"],
                                 scale=c["random_erasing"]["scale"]),
    ]
    return transforms.Compose(geo + photo + tail)


def build_datasets(data_root, size=INPUT_SIZE, robust=True,
                   merge_train_val=False):
    tr_paths, ytr = list_split(os.path.join(data_root, "classification",
                                            "train"))
    va_paths, yva = list_split(os.path.join(data_root, "classification",
                                            "validation"))
    if merge_train_val:
        tr_paths, ytr = tr_paths + va_paths, ytr + yva
    return (SteelDataset(tr_paths, ytr, train_transform(size, robust)),
            SteelDataset(va_paths, yva, eval_transform(size)))
