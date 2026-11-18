# -*- coding: utf-8 -*-
"""Stage 1 - secondary, more discriminative evaluation of the ablation runs.

The public validation split turns out to be saturated (every ablation variant
reaches a very high Macro F1), so Macro F1 alone cannot rank the four
experiments.  This script re-evaluates each checkpoint on *the same* 270
validation images after applying deterministic, label-preserving corruptions
that a real production line actually produces:

    lighting            brightness / contrast / gamma shift
    optical             Gaussian blur + resolution loss
    sensor_compression  JPEG re-compression + sensor noise
    geometric           rotation / shift the camera was never calibrated for

No extra data is used and no re-splitting happens: this is still only the
public validation split, used for model selection as the rules allow.

    python stress_eval.py --runs_dir <DIR> --data_root <STUDENT_PUBLIC>
"""
import argparse
import io
import json
import os

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import (CLASS_ORDER, INPUT_SIZE, NORM_MEAN, NORM_STD, list_split,
                    set_seed)
from evaluate import compute_metrics, predict
from model import build_model


def c_lighting(img, rng):
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.45, 1.6))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.5, 1.6))
    a = np.asarray(img, np.float32) / 255.0
    a = np.power(np.clip(a, 0, 1), rng.uniform(0.7, 1.5))
    return Image.fromarray((a * 255).astype(np.uint8))


def c_optical(img, rng):
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 2.2)))
    f = rng.uniform(0.35, 0.65)
    w, h = img.size
    return img.resize((int(w * f), int(h * f)), Image.BILINEAR).resize(
        (w, h), Image.BILINEAR)


def c_sensor(img, rng):
    buf = io.BytesIO()
    img.convert("L").save(buf, format="JPEG", quality=int(rng.integers(12, 35)))
    buf.seek(0)
    img = Image.open(buf).convert("L")
    a = np.asarray(img, np.float32) + rng.normal(0, rng.uniform(4, 14),
                                                 (img.size[1], img.size[0]))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def c_geometric(img, rng):
    img = img.rotate(float(rng.uniform(-25, 25)), resample=Image.BILINEAR,
                     fillcolor=int(np.asarray(img).mean()))
    dx, dy = int(rng.integers(-18, 19)), int(rng.integers(-18, 19))
    return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy),
                         resample=Image.BILINEAR,
                         fillcolor=int(np.asarray(img).mean()))


CORRUPTIONS = {"lighting": c_lighting, "optical": c_optical,
               "sensor_compression": c_sensor, "geometric": c_geometric}


class CorruptedVal(Dataset):
    def __init__(self, paths, labels, transform, kind=None, seed=777):
        self.paths, self.labels, self.transform = paths, labels, transform
        self.fn = CORRUPTIONS[kind] if kind else None
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")
        if self.fn is not None:
            img = self.fn(img, np.random.default_rng(self.seed + i))
        return self.transform(img), self.labels[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels = list_split(os.path.join(args.data_root, "classification",
                                            "validation"))
    tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)])

    ckpts = sorted(f for f in os.listdir(args.runs_dir)
                   if f.endswith(".pth"))
    results = {}
    for ck in ckpts:
        name = os.path.splitext(ck)[0]
        blob = torch.load(os.path.join(args.runs_dir, ck), map_location="cpu",
                          weights_only=False)
        cfg = blob.get("config", {})
        model = build_model(use_bn=cfg.get("bn", False),
                            dropout=cfg.get("dropout", 0.0))
        model.load_state_dict(blob["state_dict"])
        model.to(device).eval()

        entry = {}
        for kind in [None] + list(CORRUPTIONS):
            ds = CorruptedVal(paths, labels, tf, kind)
            ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            pred, true, _ = predict(model, ld, device)
            m = compute_metrics(true, pred)
            entry["clean" if kind is None else kind] = {
                "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                "per_class_f1": m["per_class_f1"],
                "confusion_matrix": m["confusion_matrix"]}
        entry["stress_macro_f1"] = float(np.mean(
            [entry[k]["macro_f1"] for k in CORRUPTIONS]))
        results[name] = entry
        print("{:16s} clean={:.4f} | lighting={:.4f} optical={:.4f} "
              "sensor={:.4f} geometric={:.4f} | STRESS={:.4f}".format(
                  name, entry["clean"]["macro_f1"],
                  entry["lighting"]["macro_f1"], entry["optical"]["macro_f1"],
                  entry["sensor_compression"]["macro_f1"],
                  entry["geometric"]["macro_f1"], entry["stress_macro_f1"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(results, open(args.out_json, "w", encoding="utf-8"), indent=2)
    print("saved ->", args.out_json)


if __name__ == "__main__":
    main()
