"""Stage 3 - Ground-Truth vs Prediction visualisation and dataset statistics."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import CLASS_ORDER

GT_COLOR = "#00c853"      # green  = ground truth
PRED_COLOR = "#ff1744"    # red    = prediction
CLASS_COLORS = ["#3b6ea5", "#e07b39", "#4f9d69",
                "#b5495b", "#7a5ba6", "#9c8456"]


def _draw_boxes(ax, boxes, color, label_fn, lw=1.8, ls="-"):
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, edgecolor=color, lw=lw,
                                        linestyle=ls))
        ax.text(x1 + 1, max(y1 - 3, 7), label_fn(b), color="white", fontsize=6.5,
                bbox=dict(facecolor=color, edgecolor="none", pad=0.8,
                          alpha=0.85))


def visualize_pairs(image_paths, gt, preds, out_path, score_thr=0.25,
                    ncols=5, titles=None):
    """Two rows per image block: left = GT, right = Prediction.

    ``gt``    : {image_id: {'boxes': [(cls,x1,y1,x2,y2), ...]}}
    ``preds`` : {image_id: [(cls, score, x1,y1,x2,y2), ...]}
    """
    n = len(image_paths)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows * 2, ncols,
                             figsize=(ncols * 2.5, nrows * 5.4))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")
    for i, p in enumerate(image_paths):
        iid = os.path.splitext(os.path.basename(p))[0]
        img = np.asarray(Image.open(p).convert("RGB"))
        r, c = divmod(i, ncols)
        ax_gt, ax_pr = axes[r * 2][c], axes[r * 2 + 1][c]
        for ax in (ax_gt, ax_pr):
            ax.imshow(img)
            ax.axis("off")
        _draw_boxes(ax_gt,
                    [{"bbox": b[1:], "c": b[0]} for b in gt[iid]["boxes"]],
                    GT_COLOR, lambda b: CLASS_ORDER[b["c"]])
        kept = [d for d in preds.get(iid, []) if d[1] >= score_thr]
        _draw_boxes(ax_pr,
                    [{"bbox": d[2:], "c": d[0], "s": d[1]} for d in kept],
                    PRED_COLOR,
                    lambda b: "{} {:.2f}".format(CLASS_ORDER[b["c"]], b["s"]))
        ax_gt.set_title("GT  " + iid, fontsize=7.5, color="#1b5e20")
        extra = titles.get(iid, "") if titles else ""
        ax_pr.set_title("Pred " + iid + (" " + extra if extra else ""),
                        fontsize=7.5, color="#b71c1c")
    fig.suptitle("Ground Truth (green) vs Prediction (red), conf >= {:.2f}"
                 .format(score_thr), fontsize=11, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", out_path)
    return out_path


def visualize_failures(cases, image_dir, gt, preds, out_path, score_thr=0.25,
                       ncols=5):
    """One panel per failure case, annotated with the failure type."""
    fig, axes = plt.subplots(int(np.ceil(len(cases) / ncols)), ncols,
                             figsize=(ncols * 2.7,
                                      int(np.ceil(len(cases) / ncols)) * 3.0))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for ax, case in zip(axes, cases):
        iid = case["image_id"]
        path = os.path.join(image_dir, iid + ".png")
        ax.imshow(np.asarray(Image.open(path).convert("RGB")))
        ax.axis("off")
        _draw_boxes(ax, [{"bbox": b[1:], "c": b[0]}
                         for b in gt[iid]["boxes"]], GT_COLOR,
                    lambda b: CLASS_ORDER[b["c"]], lw=1.6)
        kept = [d for d in preds.get(iid, []) if d[1] >= score_thr]
        _draw_boxes(ax, [{"bbox": d[2:], "c": d[0], "s": d[1]} for d in kept],
                    PRED_COLOR,
                    lambda b: "{} {:.2f}".format(CLASS_ORDER[b["c"]], b["s"]),
                    lw=1.6, ls="--")
        ax.set_title(case["title"], fontsize=7, color="#b71c1c")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", out_path)
    return out_path


def dataset_statistics(gt):
    """Objects per image and relative box area, per class."""
    stats = {c: {"n": 0, "rel_area": [], "aspect": []} for c in CLASS_ORDER}
    per_img = []
    for rec in gt.values():
        w, h = rec["size"]
        per_img.append(len(rec["boxes"]))
        for c, x1, y1, x2, y2 in rec["boxes"]:
            s = stats[CLASS_ORDER[c]]
            s["n"] += 1
            s["rel_area"].append((x2 - x1) * (y2 - y1) / (w * h))
            s["aspect"].append((x2 - x1) / max(y2 - y1, 1e-6))
    out = {"objects_per_image_mean": float(np.mean(per_img)),
           "objects_per_image_max": int(np.max(per_img)),
           "images": len(per_img), "per_class": {}}
    for c, s in stats.items():
        if s["n"]:
            out["per_class"][c] = {
                "n_boxes": s["n"],
                "rel_area_mean": float(np.mean(s["rel_area"])),
                "rel_area_median": float(np.median(s["rel_area"])),
                "rel_area_p10": float(np.percentile(s["rel_area"], 10)),
                "aspect_median": float(np.median(s["aspect"])),
                "small_box_ratio": float(np.mean(
                    np.array(s["rel_area"]) < 0.03)),
            }
    return out


def dataset_stats_figure(gt, out_path):
    per_class = {c: [] for c in CLASS_ORDER}
    counts = {c: 0 for c in CLASS_ORDER}
    per_img = []
    for rec in gt.values():
        w, h = rec["size"]
        per_img.append(len(rec["boxes"]))
        for c, x1, y1, x2, y2 in rec["boxes"]:
            per_class[CLASS_ORDER[c]].append((x2 - x1) * (y2 - y1) / (w * h))
            counts[CLASS_ORDER[c]] += 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(range(len(CLASS_ORDER)), [counts[c] for c in CLASS_ORDER],
                color=CLASS_COLORS)
    axes[0].set_xticks(range(len(CLASS_ORDER)),
                       [c[:9] for c in CLASS_ORDER], rotation=25, ha="right")
    axes[0].set_title("每類 bounding box 數量")
    axes[1].hist(per_img, bins=range(0, max(per_img) + 2), color="#3b6ea5",
                 align="left", rwidth=0.85)
    axes[1].set_xlabel("objects / image")
    axes[1].set_title("每張影像的物件數")
    axes[2].boxplot([per_class[c] for c in CLASS_ORDER], showfliers=False)
    axes[2].set_xticks(range(1, len(CLASS_ORDER) + 1),
                       [c[:9] for c in CLASS_ORDER])
    axes[2].set_ylabel("box area / image area")
    axes[2].set_title("相對框面積分佈")
    plt.setp(axes[2].get_xticklabels(), rotation=25, ha="right")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", out_path)
    return out_path
