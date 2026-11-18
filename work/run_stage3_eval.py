# -*- coding: utf-8 -*-
"""Stage 3 - full validation evaluation, visualisation and failure analysis.

    python run_stage3_eval.py --weights <best.pt> --data_root <STUDENT_PUBLIC>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "TeamXX_Stage3", "src"))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, rcParams
if os.path.exists(r"C:\Windows\Fonts\msjh.ttc"):
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msjh.ttc")
    rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

from common import (CLASS_ORDER, CONF_THRESHOLD, INPUT_SIZE, MAX_DET,  # noqa: E402
                    NMS_IOU_THRESHOLD, list_images, load_ground_truth,
                    set_seed)
import evaluate as EV  # noqa: E402
import visualize as VIS  # noqa: E402
from model import load_detector, model_info  # noqa: E402

FIG = os.path.join(HERE, "figures", "stage3")
OUTD = os.path.join(HERE, "outputs")


def run_predictions(model, image_paths, conf, iou, imgsz, device, batch=8):
    preds = {}
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i:i + batch]
        res = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou,
                            max_det=MAX_DET, device=device, verbose=False)
        for p, r in zip(chunk, res):
            iid = os.path.splitext(os.path.basename(p))[0]
            out = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                for j in range(len(cf)):
                    out.append((int(cl[j]), float(cf[j]), *[float(v)
                                                            for v in xyxy[j]]))
            preds[iid] = out
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--imgsz", type=int, default=INPUT_SIZE)
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--iou", type=float, default=NMS_IOU_THRESHOLD)
    ap.add_argument("--report_conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--data_yaml", default=None,
                    help="dataset yaml, only needed for the reference "
                         "Ultralytics validation")
    args = ap.parse_args()
    set_seed(42)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUTD, exist_ok=True)

    det = os.path.join(args.data_root, "detection")
    val_img = os.path.join(det, "validation", "images")
    gt = load_ground_truth(os.path.join(det, "validation", "annotations_voc"))
    paths = list_images(val_img)
    print("validation images:", len(paths), " gt images:", len(gt))

    model = load_detector(args.weights)
    info = model_info(model)

    t0 = time.time()
    preds = run_predictions(model, paths, args.conf, args.iou, args.imgsz,
                            args.device)
    batch_wall = time.time() - t0

    res = EV.evaluate_detections(gt, preds, 0.5, args.report_conf)

    # --- how the three output settings affect the official metric ---------
    setting_sweep = []
    for conf, maxdet, nms in [(0.001, 100, 0.6), (0.005, 100, 0.6),
                              (0.01, 100, 0.6), (0.05, 100, 0.6),
                              (0.001, 100, 0.5), (0.001, 100, 0.7),
                              (0.001, 30, 0.6)]:
        if (conf, maxdet, nms) == (args.conf, MAX_DET, args.iou):
            p_, r_ = preds, res
        else:
            p_ = run_predictions(model, paths, conf, nms, args.imgsz,
                                 args.device)
            p_ = {k: sorted(v, key=lambda d: -d[1])[:maxdet]
                  for k, v in p_.items()}
            r_ = EV.evaluate_detections(gt, p_, 0.5, args.report_conf)
        nb = float(np.mean([len(v) for v in p_.values()]))
        setting_sweep.append({"conf": conf, "max_det": maxdet,
                              "nms_iou": nms, "map50": r_["map50"],
                              "avg_boxes_per_image": nb,
                              "images_over_100_boxes": int(sum(
                                  1 for v in p_.values() if len(v) > 100))})
        print("  conf={:<6} max_det={:<4} nms={:<4} -> mAP {:.4f} "
              "({:.1f} boxes/img)".format(conf, maxdet, nms, r_["map50"], nb))
    sweep = EV.threshold_sweep(gt, preds)
    cutoff = EV.map_at_conf_cutoff(gt, preds)
    lat = EV.measure_inference_time(model, paths[:60], args.imgsz, args.conf,
                                    args.iou, args.device)

    # Ultralytics' own validator, for reference only.  It uses 101-point
    # interpolation and its own matching, so it does NOT agree with the TA's
    # definition; the report always quotes our own VOC-2010+ number.
    ultra = None
    try:
        vm = model.val(data=args.data_yaml, imgsz=args.imgsz, conf=0.001,
                       iou=args.iou, device=args.device, verbose=False,
                       plots=False, split="val")
        ultra = {"map50": float(vm.box.map50), "map50_95": float(vm.box.map),
                 "precision": float(vm.box.mp), "recall": float(vm.box.mr)}
        print("ultralytics built-in  mAP@0.5 = {:.4f}  (ours = {:.4f})".format(
            ultra["map50"], res["map50"]))
    except Exception as e:                                  # noqa: BLE001
        print("ultralytics val skipped:", e)

    print("\nmAP@0.5 = {:.4f}".format(res["map50"]))
    for c, v in res["per_class"].items():
        print("  {:16s} AP={:.4f}  GT={:3d}  P@{:.2f}={:.3f} R={:.3f}".format(
            c, v["ap50"], v["n_gt"], args.report_conf,
            v["precision_at_thr"], v["recall_at_thr"]))
    print("micro P/R @{:.2f}: {:.4f} / {:.4f}".format(
        args.report_conf, res["micro_precision_at_thr"],
        res["micro_recall_at_thr"]))
    print("latency: {:.1f} ms/img (single), batch wall {:.1f} ms/img".format(
        lat, batch_wall / len(paths) * 1000))

    # ------------------------------------------------------------- figures
    import matplotlib.pyplot as plt
    import plots as PL

    PL.ablation_bar([c[:9] for c in CLASS_ORDER],
                    [res["per_class"][c]["ap50"] for c in CLASS_ORDER],
                    os.path.join(FIG, "per_class_ap.png"), "AP@0.5",
                    "每一類別的 AP@0.5（mAP = {:.4f}）".format(res["map50"]),
                    baseline_idx=None, ylim=(0, 1.05))
    PL.line_plot([s["threshold"] for s in sweep],
                 {"Precision": [s["micro_precision"] for s in sweep],
                  "Recall": [s["micro_recall"] for s in sweep],
                  "F1": [s["f1"] for s in sweep]},
                 os.path.join(FIG, "threshold_sweep.png"),
                 "Confidence Threshold", "數值",
                 "Confidence Threshold 對 Precision / Recall 的影響")
    PL.line_plot([s["conf_cutoff"] for s in cutoff],
                 {"mAP@0.5": [s["map50"] for s in cutoff]},
                 os.path.join(FIG, "map_vs_cutoff.png"),
                 "輸出前先丟棄的 confidence 門檻", "mAP@0.5",
                 "為什麼提交時要保留低分框")

    # PR curves per class
    fig, ax = plt.subplots(figsize=(7, 5))
    for ci, c in enumerate(CLASS_ORDER):
        rows = sorted([(p[1], iid, p[2:]) for iid, pl in preds.items()
                       for p in pl if p[0] == ci], key=lambda r: -r[0])
        gtb = {iid: {"boxes": np.array([b[1:] for b in r["boxes"]
                                        if b[0] == ci], float),
                     "used": None} for iid, r in gt.items()}
        n_gt = sum(len(v["boxes"]) for v in gtb.values())
        for v in gtb.values():
            v["used"] = np.zeros(len(v["boxes"]), bool)
        tp = np.zeros(len(rows))
        fp = np.zeros(len(rows))
        for i, (_s, iid, box) in enumerate(rows):
            v = gtb.get(iid)
            if v is None or len(v["boxes"]) == 0:
                fp[i] = 1
                continue
            ious = EV.iou_matrix(np.array([box]), v["boxes"])[0]
            j = int(np.argmax(ious))
            if ious[j] >= 0.5 and not v["used"][j]:
                v["used"][j] = True
                tp[i] = 1
            else:
                fp[i] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        ax.plot(ctp / max(n_gt, 1), ctp / np.maximum(ctp + cfp, 1e-9),
                label="{} (AP={:.3f})".format(c, res["per_class"][c]["ap50"]),
                color=PL.PALETTE[ci], lw=1.6)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-class Precision-Recall curve @ IoU 0.5")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "pr_curves.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("fig -> pr_curves.png")

    # --------------------------------------------- GT vs Pred visualisation
    ids = sorted(gt)
    chosen, seen = [], set()
    for c in range(len(CLASS_ORDER)):          # at least one image per class
        for iid in ids:
            if iid in seen:
                continue
            if any(b[0] == c for b in gt[iid]["boxes"]):
                chosen.append(iid)
                seen.add(iid)
                break
    for iid in ids:                             # pad up to 10
        if len(chosen) >= 10:
            break
        if iid not in seen:
            chosen.append(iid)
            seen.add(iid)
    VIS.visualize_pairs([os.path.join(val_img, i + ".png") for i in chosen],
                        gt, preds, os.path.join(FIG, "gt_vs_pred.png"),
                        args.report_conf, ncols=5)

    # --------------------------------------------------------- failure cases
    cases = EV.find_failure_cases(gt, preds, 0.5, args.report_conf)
    picked = []
    for kind, title in [("false_negative", "FN 漏檢"),
                        ("false_positive", "FP 誤檢"),
                        ("poor_localization", "Poor Localization"),
                        ("class_confusion", "類別混淆")]:
        for c in cases[kind][:3]:
            lab = CLASS_ORDER[int(c.get("gt", c.get("pred"))[0])]
            extra = (" IoU={:.2f}".format(c["iou"]) if "iou" in c else "")
            picked.append({"image_id": c["image_id"],
                           "title": "{}｜{}{}".format(title, lab, extra),
                           "kind": kind, "detail": c})
    picked = picked[:12]
    if picked:
        VIS.visualize_failures(picked, val_img, gt, preds,
                               os.path.join(FIG, "failures.png"),
                               args.report_conf, ncols=4)

    def jsonable(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (list, tuple)):
            return [jsonable(x) for x in o]
        if isinstance(o, dict):
            return {k: jsonable(v) for k, v in o.items()}
        return o

    summary = {
        "weights": os.path.abspath(args.weights),
        "imgsz": args.imgsz, "conf_threshold": args.conf,
        "nms_iou_threshold": args.iou, "max_det": MAX_DET,
        "report_conf": args.report_conf,
        "model": info,
        "checkpoint_mb": os.path.getsize(args.weights) / 1024 ** 2,
        "latency_ms_single": lat,
        "latency_ms_batched": batch_wall / len(paths) * 1000,
        "validation": jsonable(res),
        "ultralytics_reference": ultra,
        "threshold_sweep": jsonable(sweep),
        "map_vs_conf_cutoff": jsonable(cutoff),
        "setting_sweep": jsonable(setting_sweep),
        "failure_counts": {k: len(v) for k, v in cases.items()},
        "failure_examples": jsonable(picked),
        "visualized_ids": chosen,
    }
    p = os.path.join(OUTD, "stage3_val_results.json")
    json.dump(summary, open(p, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("saved ->", p)
    print("failure counts:", summary["failure_counts"])


if __name__ == "__main__":
    main()
