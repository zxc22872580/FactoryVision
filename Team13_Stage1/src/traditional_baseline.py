"""Stage 1 / Part A - Traditional (non-deep) baselines: PCA + SVM and PCA + KNN.

Pipeline (identical for both classifiers, so the comparison is fair):
    grayscale -> resize 64x64 -> flatten (4096-d) -> StandardScaler
    -> PCA(n_components) -> classifier

Usage:
    python traditional_baseline.py --data_root <STUDENT_PUBLIC> --out_dir <DIR>
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CLASS_ORDER, SEED, list_split, set_seed

FEAT_SIZE = 64


def load_features(paths):
    X = np.zeros((len(paths), FEAT_SIZE * FEAT_SIZE), dtype=np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L").resize((FEAT_SIZE, FEAT_SIZE),
                                               Image.BILINEAR)
        X[i] = np.asarray(im, dtype=np.float32).ravel() / 255.0
    return X


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred,
                                                 average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred,
                                           average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, average=None,
                                 labels=list(range(len(CLASS_ORDER))),
                                 zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASS_ORDER)))).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    set_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    tr_dir = os.path.join(args.data_root, "classification", "train")
    va_dir = os.path.join(args.data_root, "classification", "validation")
    tr_paths, ytr = list_split(tr_dir)
    va_paths, yva = list_split(va_dir)
    ytr, yva = np.array(ytr), np.array(yva)

    t0 = time.time()
    Xtr, Xva = load_features(tr_paths), load_features(va_paths)
    print(f"features: train {Xtr.shape} val {Xva.shape} "
          f"({time.time() - t0:.1f}s)")

    results = {}

    # ---- hyper-parameter selection on the validation split --------------
    grid_svm = [{"pca": n, "C": c, "gamma": g}
                for n in (64, 128, 192)
                for c in (1.0, 10.0, 100.0)
                for g in ("scale", 0.01, 0.001)]
    grid_knn = [{"pca": n, "k": k, "weights": w}
                for n in (64, 128, 192)
                for k in (1, 3, 5, 7)
                for w in ("uniform", "distance")]

    def build(kind, hp):
        steps = [("scaler", StandardScaler()),
                 ("pca", PCA(n_components=hp["pca"], random_state=SEED,
                             whiten=True))]
        if kind == "svm":
            steps.append(("clf", SVC(C=hp["C"], gamma=hp["gamma"],
                                     kernel="rbf", random_state=SEED)))
        else:
            steps.append(("clf", KNeighborsClassifier(
                n_neighbors=hp["k"], weights=hp["weights"])))
        return Pipeline(steps)

    for kind, grid in (("svm", grid_svm), ("knn", grid_knn)):
        best, best_f1, search = None, -1.0, []
        for hp in grid:
            pipe = build(kind, hp)
            pipe.fit(Xtr, ytr)
            f1 = f1_score(yva, pipe.predict(Xva), average="macro",
                          zero_division=0)
            search.append({**hp, "val_macro_f1": float(f1)})
            if f1 > best_f1:
                best_f1, best = f1, hp
        pipe = build(kind, best)
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        fit_s = time.time() - t0
        t0 = time.time()
        pred = pipe.predict(Xva)
        infer_ms = (time.time() - t0) / len(yva) * 1000
        evr = float(pipe.named_steps["pca"].explained_variance_ratio_.sum())
        results[f"pca_{kind}"] = {
            "best_hparams": best,
            "pca_explained_variance_ratio": evr,
            "fit_seconds": fit_s,
            "val_inference_ms_per_image": infer_ms,
            "val": metrics(yva, pred),
            "train": metrics(ytr, pipe.predict(Xtr)),
            "search": search,
        }
        print(f"\n=== PCA + {kind.upper()} ===")
        print("best:", best, f"(PCA EVR={evr:.3f})")
        print(classification_report(yva, pred, target_names=CLASS_ORDER,
                                    digits=4, zero_division=0))

    out = os.path.join(args.out_dir, "stage1_traditional_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
