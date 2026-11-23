# -*- coding: utf-8 -*-
"""Diagnose why the public validation split saturates.

Computes, for the official (fixed) classification split:
  * exact duplicates (MD5) inside train and between train and validation;
  * near-duplicates, via the correlation of 32x32 mean/std-normalised images;
  * the distribution of each validation image's maximum similarity to train.

The split itself is never modified - the course requires the fixed split; this
is purely a diagnostic that explains the validation numbers in the report.
"""
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Team13_Stage1", "src"))
sys.path.insert(0, HERE)
from common import CLASS_ORDER, list_split  # noqa: E402
import plots as PL  # noqa: E402

DATA = os.environ.get(
    "FACTORYVISION_DATA",
    r"C:\Users\zxc22\OneDrive\桌面\中興大學\電腦視覺\期末"
    r"\FactoryVision_STUDENT_PUBLIC\STUDENT_PUBLIC")


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def feat(p, n=32):
    a = np.asarray(Image.open(p).convert("L").resize((n, n)), np.float32)
    return ((a - a.mean()) / (a.std() + 1e-8)).ravel()


def main():
    cls = os.path.join(DATA, "classification")
    tr, ytr = list_split(os.path.join(cls, "train"))
    va, yva = list_split(os.path.join(cls, "validation"))

    htr = {}
    for p in tr:
        htr.setdefault(md5(p), []).append(p)
    exact = [(os.path.basename(p), os.path.basename(htr[md5(p)][0]))
             for p in va if md5(p) in htr]

    Ftr = np.stack([feat(p) for p in tr])
    Fva = np.stack([feat(p) for p in va])
    sim = (Fva @ Ftr.T) / Ftr.shape[1]
    mx = sim.max(1)
    arg = sim.argmax(1)
    same_class = np.array([ytr[a] == y for a, y in zip(arg, yva)])

    per_class = {}
    for c in range(len(CLASS_ORDER)):
        m = np.array(yva) == c
        per_class[CLASS_ORDER[c]] = {
            "max_corr_mean": float(mx[m].mean()),
            "n_above_0.95": int((mx[m] > 0.95).sum()),
            "n_above_0.90": int((mx[m] > 0.90).sum()),
        }

    res = {
        "n_train": len(tr), "n_val": len(va),
        "unique_train_hashes": len(htr),
        "exact_train_val_duplicates": exact,
        "near_dup_corr_gt_0.95": int((mx > 0.95).sum()),
        "near_dup_corr_gt_0.90": int((mx > 0.90).sum()),
        "max_corr_mean": float(mx.mean()),
        "max_corr_median": float(np.median(mx)),
        "max_corr_p95": float(np.percentile(mx, 95)),
        "nearest_train_neighbour_same_class_ratio": float(same_class.mean()),
        "per_class": per_class,
    }
    out = os.path.join(HERE, "outputs", "split_analysis.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"},
                     indent=2, ensure_ascii=False))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(mx, bins=40, color="#3b6ea5")
    ax[0].axvline(0.95, color="#b5495b", ls="--",
                  label="corr = 0.95（高度相似）")
    ax[0].set_xlabel("每張 validation 影像對訓練集的最大相似度")
    ax[0].set_ylabel("張數")
    ax[0].set_title("Validation 與 Train 的相似度分佈")
    ax[0].legend(fontsize=8)
    vals = [per_class[c]["n_above_0.95"] for c in CLASS_ORDER]
    ax[1].bar(range(6), vals, color=PL.PALETTE)
    ax[1].set_xticks(range(6), PL.SHORT, rotation=25, ha="right")
    ax[1].set_ylabel("corr > 0.95 的張數 (/45)")
    ax[1].set_title("各類別的近重複張數")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(HERE, "figures", "split_similarity.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("fig ->", p)


if __name__ == "__main__":
    main()
