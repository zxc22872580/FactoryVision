"""Stage 3 - evaluation that reproduces the TA runner's mAP@0.5 definition.

Definition implemented here (identical to the specification):
    * IoU threshold fixed at 0.5;
    * per class, predictions sorted by confidence (descending);
    * one ground-truth box can be matched by at most one prediction, any later
      duplicate on the same GT counts as a False Positive;
    * wrong class, IoU < 0.5 or an already-matched GT  ->  False Positive;
    * per-class AP with Pascal VOC 2010+ all-points interpolation;
    * mAP@0.5 = unweighted mean of the six class APs.
"""
import json
import time

import numpy as np

from common import CLASS_ORDER, NUM_CLASSES


def iou_matrix(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-12)).astype(np.float32)


def voc_ap_all_points(recall, precision):
    """Pascal VOC 2010+ all-points interpolated AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):          # precision envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_detections(gt, preds, iou_thr=0.5, score_thr_for_pr=0.25):
    """gt    : {image_id: {'boxes': [(cls,x1,y1,x2,y2), ...]}}
       preds : {image_id: [(cls, score, x1,y1,x2,y2), ...]}
    Returns a dict with per-class AP, mAP@0.5 and PR at a fixed threshold."""
    per_class = {}
    aps = []
    for c in range(NUM_CLASSES):
        # ---- gather ground truth of this class ---------------------------
        gt_by_img, n_gt = {}, 0
        for iid, rec in gt.items():
            boxes = [b[1:] for b in rec["boxes"] if b[0] == c]
            if boxes:
                gt_by_img[iid] = {"boxes": np.array(boxes, np.float64),
                                  "used": np.zeros(len(boxes), bool)}
                n_gt += len(boxes)
        # ---- gather and rank predictions of this class -------------------
        rows = [(iid, p[1], p[2:]) for iid, plist in preds.items()
                for p in plist if p[0] == c]
        rows.sort(key=lambda r: -r[1])
        tp = np.zeros(len(rows), np.float64)
        fp = np.zeros(len(rows), np.float64)
        for i, (iid, _score, box) in enumerate(rows):
            rec = gt_by_img.get(iid)
            if rec is None:
                fp[i] = 1
                continue
            ious = iou_matrix(np.array([box], np.float64), rec["boxes"])[0]
            j = int(np.argmax(ious)) if len(ious) else -1
            if j >= 0 and ious[j] >= iou_thr and not rec["used"][j]:
                rec["used"][j] = True
                tp[i] = 1
            else:
                fp[i] = 1
        if n_gt == 0:
            ap = 0.0
            recall = precision = np.array([])
        else:
            ctp, cfp = np.cumsum(tp), np.cumsum(fp)
            recall = ctp / n_gt
            precision = ctp / np.maximum(ctp + cfp, 1e-12)
            ap = voc_ap_all_points(recall, precision)
        # ---- precision / recall at a reporting threshold -----------------
        keep = [i for i, r in enumerate(rows) if r[1] >= score_thr_for_pr]
        tp_t, fp_t = float(tp[keep].sum()), float(fp[keep].sum())
        per_class[CLASS_ORDER[c]] = {
            "ap50": ap, "n_gt": n_gt, "n_pred": len(rows),
            "tp_at_thr": tp_t, "fp_at_thr": fp_t,
            "fn_at_thr": n_gt - tp_t,
            "precision_at_thr": tp_t / max(tp_t + fp_t, 1e-12),
            "recall_at_thr": tp_t / max(n_gt, 1e-12),
        }
        aps.append(ap)

    tot_tp = sum(v["tp_at_thr"] for v in per_class.values())
    tot_fp = sum(v["fp_at_thr"] for v in per_class.values())
    tot_gt = sum(v["n_gt"] for v in per_class.values())
    return {
        "iou_threshold": iou_thr,
        "score_threshold_for_pr": score_thr_for_pr,
        "map50": float(np.mean(aps)),
        "per_class": per_class,
        "macro_precision_at_thr": float(np.mean(
            [v["precision_at_thr"] for v in per_class.values()])),
        "macro_recall_at_thr": float(np.mean(
            [v["recall_at_thr"] for v in per_class.values()])),
        "micro_precision_at_thr": tot_tp / max(tot_tp + tot_fp, 1e-12),
        "micro_recall_at_thr": tot_tp / max(tot_gt, 1e-12),
        "n_gt_total": tot_gt,
    }


def threshold_sweep(gt, preds, thresholds=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                           0.40, 0.50, 0.60, 0.70)):
    """How Precision / Recall move with the confidence threshold (mAP does not
    depend on it as long as low-score boxes are kept)."""
    out = []
    for t in thresholds:
        r = evaluate_detections(gt, preds, score_thr_for_pr=t)
        out.append({"threshold": t,
                    "micro_precision": r["micro_precision_at_thr"],
                    "micro_recall": r["micro_recall_at_thr"],
                    "macro_precision": r["macro_precision_at_thr"],
                    "macro_recall": r["macro_recall_at_thr"],
                    "f1": (2 * r["micro_precision_at_thr"]
                           * r["micro_recall_at_thr"]
                           / max(r["micro_precision_at_thr"]
                                 + r["micro_recall_at_thr"], 1e-12))})
    return out


def map_at_conf_cutoff(gt, preds, cutoffs=(0.001, 0.01, 0.05, 0.10, 0.25)):
    """mAP@0.5 when predictions below a cutoff are *discarded* before scoring.
    Shows why the submitted detector must keep low-confidence boxes."""
    out = []
    for c in cutoffs:
        filtered = {k: [p for p in v if p[1] >= c] for k, v in preds.items()}
        out.append({"conf_cutoff": c,
                    "map50": evaluate_detections(gt, filtered)["map50"],
                    "n_boxes": sum(len(v) for v in filtered.values())})
    return out


def find_failure_cases(gt, preds, iou_thr=0.5, score_thr=0.25):
    """Classify each prediction / GT into TP, FP, FN or poor localization."""
    cases = {"false_positive": [], "false_negative": [],
             "poor_localization": [], "class_confusion": []}
    for iid, rec in gt.items():
        gboxes = rec["boxes"]
        pboxes = [p for p in preds.get(iid, []) if p[1] >= score_thr]
        used = [False] * len(gboxes)
        for p in sorted(pboxes, key=lambda r: -r[1]):
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gboxes):
                if used[j]:
                    continue
                v = iou_matrix(np.array([p[2:]]), np.array([g[1:]]))[0, 0]
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j < 0:
                cases["false_positive"].append(
                    {"image_id": iid, "pred": p, "iou": 0.0})
                continue
            g = gboxes[best_j]
            if best_iou >= iou_thr and g[0] == p[0]:
                used[best_j] = True
            elif g[0] != p[0] and best_iou >= 0.3:
                cases["class_confusion"].append(
                    {"image_id": iid, "pred": p, "gt": g, "iou": float(best_iou)})
            elif 0.1 <= best_iou < iou_thr:
                cases["poor_localization"].append(
                    {"image_id": iid, "pred": p, "gt": g, "iou": float(best_iou)})
            else:
                cases["false_positive"].append(
                    {"image_id": iid, "pred": p, "iou": float(best_iou)})
        for j, g in enumerate(gboxes):
            if not used[j]:
                cases["false_negative"].append({"image_id": iid, "gt": g})
    return cases


def load_detections_json(path):
    preds = {}
    for rec in json.load(open(path, encoding="utf-8")):
        preds[rec["image_id"]] = [
            (CLASS_ORDER.index(d["class"]), float(d["confidence"]),
             *[float(v) for v in d["bbox"]]) for d in rec["detections"]]
    return preds


def measure_inference_time(model, image_paths, imgsz=640, conf=0.01,
                           iou=0.6, device=0, warmup=5, repeats=1):
    """End-to-end wall-clock time per image (pre + forward + NMS)."""
    for p in image_paths[:warmup]:
        model.predict(p, imgsz=imgsz, conf=conf, iou=iou, device=device,
                      verbose=False)
    t0 = time.time()
    n = 0
    for _ in range(repeats):
        for p in image_paths:
            model.predict(p, imgsz=imgsz, conf=conf, iou=iou, device=device,
                          verbose=False)
            n += 1
    return (time.time() - t0) / n * 1000.0
