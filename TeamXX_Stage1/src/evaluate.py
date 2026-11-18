"""Stage 1 - evaluation utilities (metrics, confusion matrix, curves)."""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from common import CLASS_ORDER, NUM_CLASSES


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logit = model(x)
        probs.append(torch.softmax(logit, 1).cpu().numpy())
        preds.append(logit.argmax(1).cpu().numpy())
        trues.append(np.asarray(y))
    return (np.concatenate(preds), np.concatenate(trues),
            np.concatenate(probs))


def compute_metrics(y_true, y_pred):
    labels = list(range(NUM_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels,
                                                 average="macro",
                                                 zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels,
                                           average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "per_class_precision": precision_score(y_true, y_pred, labels=labels,
                                               average=None,
                                               zero_division=0).tolist(),
        "per_class_recall": recall_score(y_true, y_pred, labels=labels,
                                         average=None,
                                         zero_division=0).tolist(),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None,
                                 zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=labels).tolist(),
    }


@torch.no_grad()
def measure_latency(model, device, input_size=128, in_ch=1, n=100, bs=1):
    model.eval()
    x = torch.randn(bs, in_ch, input_size, input_size, device=device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


def evaluate_checkpoint(ckpt_path, data_root, device=None, batch_size=64):
    """Rebuild the model from a checkpoint and report validation metrics."""
    from torch.utils.data import DataLoader

    from dataset import build_datasets
    from model import build_model, count_parameters

    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model(use_bn=cfg.get("bn", False),
                        dropout=cfg.get("dropout", 0.0)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    _, val_ds = build_datasets(data_root, augment=False,
                               size=cfg.get("input_size", 128))
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    pred, true, _ = predict(model, loader, device)
    m = compute_metrics(true, pred)
    total, trainable = count_parameters(model)
    m["params"], m["trainable_params"] = total, trainable
    m["checkpoint_mb"] = os.path.getsize(ckpt_path) / 1024 ** 2
    m["latency_ms_bs1"] = measure_latency(model, device,
                                          cfg.get("input_size", 128))
    return m, ckpt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()
    m, _ = evaluate_checkpoint(a.checkpoint, a.data_root)
    print(json.dumps({k: v for k, v in m.items()
                      if k != "confusion_matrix"}, indent=2))
    print("confusion matrix:")
    print(np.array(m["confusion_matrix"]))
    if a.out_json:
        json.dump(m, open(a.out_json, "w"), indent=2)
