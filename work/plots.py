"""Figure generation shared by all three stages and the final report."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams

CJK = r"C:\Windows\Fonts\msjh.ttc"
if os.path.exists(CJK):
    font_manager.fontManager.addfont(CJK)
    rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 150
rcParams["savefig.bbox"] = "tight"
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

CLASS_ORDER = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled_in_scale", "scratches"]
SHORT = ["crazing", "inclusion", "patches", "pitted", "rolled", "scratches"]
PALETTE = ["#3b6ea5", "#e07b39", "#4f9d69", "#b5495b", "#7a5ba6", "#9c8456"]


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("fig ->", path)
    return path


# --------------------------------------------------------------------------
def training_curves(summaries, labels, out_path, title=""):
    """2x2 panel: train/val loss, train/val accuracy, val macro F1."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for s, lab, col in zip(summaries, labels, PALETTE):
        h = s["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], color=col, ls="--", lw=1.2)
        axes[0].plot(ep, h["val_loss"], color=col, lw=1.6, label=lab)
        axes[1].plot(ep, h["train_acc"], color=col, ls="--", lw=1.2)
        axes[1].plot(ep, h["val_acc"], color=col, lw=1.6, label=lab)
        axes[2].plot(ep, h["val_macro_f1"], color=col, lw=1.6, label=lab)
        axes[2].scatter([s["best_epoch"]],
                        [h["val_macro_f1"][s["best_epoch"] - 1]],
                        color=col, s=36, zorder=5, marker="*")
    axes[0].set_title("Loss (虛線=train, 實線=val)")
    axes[1].set_title("Accuracy (虛線=train, 實線=val)")
    axes[2].set_title("Validation Macro F1 (★ = 選用 epoch)")
    for a in axes:
        a.set_xlabel("epoch")
        a.legend(fontsize=8)
    if title:
        fig.suptitle(title, y=1.03, fontsize=13)
    return save(fig, out_path)


def confusion_matrix_fig(cm, out_path, title="Confusion Matrix",
                         names=None, normalize=False):
    cm = np.array(cm, dtype=float)
    names = names or SHORT
    disp = cm / np.maximum(cm.sum(1, keepdims=True), 1e-9) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(disp, cmap="Blues", vmin=0,
                   vmax=1 if normalize else disp.max())
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    thr = disp.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = "{:.2f}".format(disp[i, j]) if normalize else "{:d}".format(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if disp[i, j] > thr else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return save(fig, out_path)


def ablation_bar(labels, values, out_path, ylabel="Validation Macro F1",
                 title="", baseline_idx=0, extra=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=PALETTE[:len(labels)], width=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                "{:.4f}".format(v), ha="center", fontsize=9)
    if baseline_idx is not None:
        ax.axhline(values[baseline_idx], color="#888", ls=":", lw=1.2,
                   label="baseline ({:.4f})".format(values[baseline_idx]))
        ax.legend(fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim or (min(values) - 0.08, min(1.02, max(values) + 0.06)))
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    if extra:
        ax.text(0.99, 0.02, extra, transform=ax.transAxes, ha="right",
                fontsize=8, color="#555")
    return save(fig, out_path)


def per_class_f1_bar(series, labels, out_path, title=""):
    """series: list of per-class F1 lists."""
    x = np.arange(len(SHORT))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (s, lab) in enumerate(zip(series, labels)):
        ax.bar(x + i * w - 0.4 + w / 2, s, width=w, label=lab,
               color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x, SHORT, rotation=20, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=min(len(series), 4))
    return save(fig, out_path)


def grouped_bar(categories, groups, out_path, ylabel="", title="",
                annotate=True, ylim=None):
    """groups: dict label -> list of values aligned with categories."""
    x = np.arange(len(categories))
    w = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, (lab, vals) in enumerate(groups.items()):
        b = ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=lab,
                   color=PALETTE[i % len(PALETTE)])
        if annotate:
            for r, v in zip(b, vals):
                ax.text(r.get_x() + r.get_width() / 2, v + 0.004,
                        "{:.3f}".format(v), ha="center", fontsize=7.5)
    ax.set_xticks(x, categories, rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    return save(fig, out_path)


def scatter_tradeoff(points, out_path, xlabel="", ylabel="", title=""):
    """points: list of (x, y, label)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, (x, y, lab) in enumerate(points):
        ax.scatter(x, y, s=90, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8, 5),
                    fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return save(fig, out_path)


def line_plot(xs, series, out_path, xlabel="", ylabel="", title=""):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (lab, ys) in enumerate(series.items()):
        ax.plot(xs, ys, marker="o", ms=4, lw=1.6, label=lab,
                color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return save(fig, out_path)
