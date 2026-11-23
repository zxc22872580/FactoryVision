"""Stage 2 - metrics, latency measurement and robustness evaluation.

The robustness evaluation reproduces, on the *public* validation split, the
three corruption families the TA uses for the hidden robustness benchmark
(lighting / optical / sensor_compression).  It is only an internal estimate:
the submitted model and preprocessing are identical for all of them.
"""
import io
import os
import time

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from common import NUM_CLASSES


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True))
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(np.asarray(y))
    return np.concatenate(preds), np.concatenate(trues)


def compute_metrics(y_true, y_pred):
    labels = list(range(NUM_CLASSES))
    kw = dict(labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred,
                                                 average="macro", **kw)),
        "macro_recall": float(recall_score(y_true, y_pred,
                                           average="macro", **kw)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, **kw).tolist(),
        "per_class_precision": precision_score(y_true, y_pred, average=None,
                                               **kw).tolist(),
        "per_class_recall": recall_score(y_true, y_pred, average=None,
                                         **kw).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=labels).tolist(),
    }


@torch.no_grad()
def measure_latency(model, device, input_size=224, n=50, bs=1, warmup=10):
    model.eval()
    x = torch.randn(bs, 3, input_size, input_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


# --------------------------------------------------------------------------
# deterministic corruption generators used for the internal robustness probe
# --------------------------------------------------------------------------
def corrupt_lighting(img, rng):
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.45, 1.65))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.5, 1.6))
    return img


def corrupt_optical(img, rng):
    from PIL import ImageFilter
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.4)))
    f = rng.uniform(0.35, 0.6)
    w, h = img.size
    img = img.resize((int(w * f), int(h * f)), Image.BILINEAR)
    return img.resize((w, h), Image.BILINEAR)


def corrupt_sensor(img, rng):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(rng.integers(12, 35)))
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    a = np.asarray(img).astype(np.float32)
    a = a + rng.normal(0, rng.uniform(4, 14), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


CORRUPTIONS = {"lighting": corrupt_lighting,
               "optical": corrupt_optical,
               "sensor_compression": corrupt_sensor}


class CorruptedDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform, kind, seed=1234):
        self.paths, self.labels, self.transform = paths, labels, transform
        self.fn = CORRUPTIONS[kind]
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        img = Image.open(self.paths[i]).convert("L").convert("RGB")
        return self.transform(self.fn(img, rng)), self.labels[i]


def robustness_report(model, paths, labels, transform, device, batch_size=32):
    from torch.utils.data import DataLoader
    out = {}
    for kind in CORRUPTIONS:
        ds = CorruptedDataset(paths, labels, transform, kind)
        ld = DataLoader(ds, batch_size=batch_size, shuffle=False)
        p, t = predict(model, ld, device)
        out[kind] = compute_metrics(t, p)
    out["robust_macro_f1"] = float(np.mean(
        [out[k]["macro_f1"] for k in CORRUPTIONS]))
    return out


def checkpoint_size_mb(path):
    return os.path.getsize(path) / 1024 ** 2
