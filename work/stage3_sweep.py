# -*- coding: utf-8 -*-
"""Stage 3 - detector / hyper-parameter sweep (local, 4 GB GPU friendly).

Every run uses the same seed, the same dataset folder and the same epoch
budget unless the experiment says otherwise, then is scored with *our own*
VOC2010+ mAP implementation -- the one that matches the TA's definition -- so
the numbers are directly comparable to the 0.7419 the currently submitted
YOLOv8s checkpoint reaches.

    python stage3_sweep.py --list
    python stage3_sweep.py --only yolo11s_512,yolov9s_512 --epochs 100
    python stage3_sweep.py --table

Results are appended to outputs/stage3_sweep.json after EVERY run, so a crash
or a reboot never loses finished work; re-running a recorded name is skipped
unless --force is given.

Measured on a GTX 1650 Ti 4 GB (2-epoch probe, see probe_speed.py):

    yolov8s  b8  @640    99.6 s/epoch   peak 3970 MB   <- at the VRAM ceiling
    yolo11s  b4  @640   105.1 s/epoch   peak 2655 MB
    yolo11s  b8  @512    73.0 s/epoch   peak 3381 MB   <- best time/memory
    yolo11s  b16 @416    56.2 s/epoch   peak 3817 MB

Anything whose peak approaches 4000 MB spills into shared system memory over
WDDM and roughly halves throughput, so a smaller imgsz can be the faster AND
the more accurate choice. The source images are only 200x200, so 640 is
already a 3.2x upscale.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Team13_Stage3", "src"))

from common import (CLASS_ORDER, CONF_THRESHOLD, MAX_DET,  # noqa: E402
                    NMS_IOU_THRESHOLD, list_images, load_ground_truth,
                    set_seed)
import evaluate as EV  # noqa: E402

DS = os.path.join(HERE, "det_dataset")
YAML = os.path.join(DS, "factoryvision.yaml")
RUNS = os.path.join(HERE, "stage3_sweep")
OUT = os.path.join(HERE, "outputs", "stage3_sweep.json")

# The steel images are grayscale, so hue/saturation jitter is pointless;
# only brightness (hsv_v) is kept. Same augmentation as the submitted run.
BASE_AUG = dict(degrees=10.0, translate=0.1, scale=0.5, fliplr=0.5,
                flipud=0.5, mosaic=1.0, mixup=0.0,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.4)

EXPERIMENTS = {
    # ---- control: reproduces the submitted checkpoint's recipe ------------
    "yolov8s_640": dict(weights="yolov8s.pt", batch=8, imgsz=640),

    # ---- architecture screening at the fast setting ----------------------
    "yolo11s_512": dict(weights="yolo11s.pt", batch=8, imgsz=512),
    "yolov9s_512": dict(weights="yolov9s.pt", batch=8, imgsz=512),
    "yolov10s_512": dict(weights="yolov10s.pt", batch=8, imgsz=512),
    "yolo12s_512": dict(weights="yolo12s.pt", batch=8, imgsz=512),
    "yolov8s_512": dict(weights="yolov8s.pt", batch=8, imgsz=512),

    # ---- does the 640 upscale actually buy anything? ---------------------
    "yolo11s_640": dict(weights="yolo11s.pt", batch=4, imgsz=640),
    "yolo11s_416": dict(weights="yolo11s.pt", batch=16, imgsz=416),

    # ---- the cheap win: the submitted run was still improving at ep 100 --
    "yolov8s_640_long": dict(weights="yolov8s.pt", batch=8, imgsz=640,
                             epochs=250, close_mosaic=40),
    "yolo11s_512_long": dict(weights="yolo11s.pt", batch=8, imgsz=512,
                             epochs=250, close_mosaic=40),

    # ---- milder geometry: crazing/rolled_in_scale boxes are diffuse ------
    "yolo11s_512_mildaug": dict(weights="yolo11s.pt", batch=8, imgsz=512,
                                aug=dict(scale=0.25, degrees=5.0)),
    "yolo11s_512_cos": dict(weights="yolo11s.pt", batch=8, imgsz=512,
                            cos_lr=True, lr0=0.01, lrf=0.05),

    # ---- bigger backbone, only if the VRAM holds -------------------------
    "yolo11m_512": dict(weights="yolo11m.pt", batch=4, imgsz=512),
}

# Suggested orders for a single unattended session.
PRESETS = {
    "screen": ["yolov8s_512", "yolo11s_512", "yolov9s_512", "yolov10s_512",
               "yolo12s_512"],
    "resolution": ["yolo11s_416", "yolo11s_512", "yolo11s_640"],
    "long": ["yolov8s_640_long", "yolo11s_512_long"],
    "all": list(EXPERIMENTS),
}


def val_paths(data_root):
    img_dir = os.path.join(data_root, "detection", "validation", "images")
    xml_dir = os.path.join(data_root, "detection", "validation",
                           "annotations_voc")
    return img_dir, xml_dir


def score(weights, imgsz, data_root):
    """Our own TA-matching mAP@0.5 on the public validation split."""
    from ultralytics import YOLO
    set_seed(42)
    model = YOLO(weights)
    dev = 0 if (torch.cuda.is_available() and torch.cuda.device_count()) else "cpu"
    img_dir, xml_dir = val_paths(data_root)
    paths = list_images(img_dir)
    gt = load_ground_truth(xml_dir)
    if not paths:
        raise RuntimeError("no validation images under " + img_dir)

    preds = {}
    t0 = time.time()
    for i in range(0, len(paths), 8):
        chunk = paths[i:i + 8]
        res = model.predict(chunk, imgsz=imgsz, conf=CONF_THRESHOLD,
                            iou=NMS_IOU_THRESHOLD, max_det=MAX_DET,
                            device=dev, verbose=False)
        for p, r in zip(chunk, res):
            iid = os.path.splitext(os.path.basename(p))[0]
            out = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                for j in range(len(cf)):
                    out.append((int(cl[j]), float(cf[j]),
                                *[float(v) for v in xyxy[j]]))
            preds[iid] = out
    ms = (time.time() - t0) / len(paths) * 1000.0
    return EV.evaluate_detections(gt, preds, iou_thr=0.5), ms


def run_one(name, cfg, default_epochs, data_root):
    from ultralytics import YOLO
    import csv

    aug = dict(BASE_AUG)
    aug.update(cfg.get("aug", {}))
    epochs = cfg.get("epochs", default_epochs)
    kw = dict(data=YAML, epochs=epochs, batch=cfg["batch"], imgsz=cfg["imgsz"],
              seed=42, deterministic=True, project=RUNS, name=name,
              exist_ok=True, workers=4, cache=True,
              patience=epochs,                 # never early-stop a sweep run
              close_mosaic=cfg.get("close_mosaic", 20),
              lr0=cfg.get("lr0", 0.01), lrf=cfg.get("lrf", 0.01),
              cos_lr=cfg.get("cos_lr", False),
              verbose=False, plots=False, **aug)

    set_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    YOLO(os.path.join(HERE, cfg["weights"])).train(**kw)
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_reserved() / 1e6

    # per-epoch cost, straight out of Ultralytics' own log
    sec_per_epoch = None
    csv_path = os.path.join(RUNS, name, "results.csv")
    if os.path.exists(csv_path):
        t = [float(r["time"]) for r in csv.DictReader(open(csv_path))]
        if len(t) > 1:
            sec_per_epoch = (t[-1] - t[0]) / (len(t) - 1)

    best = os.path.join(RUNS, name, "weights", "best.pt")
    res, ms = score(best, cfg["imgsz"], data_root)
    return {"name": name, "weights": cfg["weights"], "epochs": epochs,
            "batch": cfg["batch"], "imgsz": cfg["imgsz"],
            "close_mosaic": kw["close_mosaic"], "cos_lr": kw["cos_lr"],
            "lr0": kw["lr0"], "lrf": kw["lrf"], "aug": aug,
            "train_seconds": train_s, "sec_per_epoch": sec_per_epoch,
            "peak_vram_mb": peak, "latency_ms": ms,
            "map50_ours": res["map50"],
            "per_class_ap50": {k: v["ap50"]
                               for k, v in res["per_class"].items()},
            "per_class_full": res["per_class"],
            "best_weights": best,
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"}


def load_all():
    if os.path.exists(OUT):
        return json.load(open(OUT, encoding="utf-8"))
    return []


def save_all(recs):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1, ensure_ascii=False)


SUBMITTED = 0.7419211054363365   # the checkpoint currently in Team13_Stage3


def table(recs):
    recs = sorted(recs, key=lambda r: -r["map50_ours"])
    w = max([len(r["name"]) for r in recs] + [9])
    head = ("%-*s %4s %3s %4s | %7s %7s | " % (w, "run", "ep", "bs", "img",
                                               "mAP@.5", "vs now")
            + " ".join("%-6s" % c[:6] for c in CLASS_ORDER)
            + " | %5s %5s" % ("ms", "VRAM"))
    print(head)
    print("-" * len(head))
    for r in recs:
        print("%-*s %4d %3d %4d | %7.4f %+7.4f | "
              % (w, r["name"], r["epochs"], r["batch"], r["imgsz"],
                 r["map50_ours"], r["map50_ours"] - SUBMITTED)
              + " ".join("%-6.4f" % r["per_class_ap50"][c]
                         for c in CLASS_ORDER)
              + " | %5.1f %5.0f" % (r["latency_ms"], r["peak_vram_mb"]))
    print()
    print("vs now = 跟目前繳交的 YOLOv8s checkpoint (%.4f) 的差距" % SUBMITTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get(
        "FACTORYVISION_DATA",
        r"C:\Users\zxc22\OneDrive\桌面\中興大學\電腦視覺\期末"
        r"\FactoryVision_STUDENT_PUBLIC\STUDENT_PUBLIC"))
    ap.add_argument("--only", default=None,
                    help="comma separated experiment names, or a preset: "
                         + ", ".join(PRESETS))
    ap.add_argument("--epochs", type=int, default=100,
                    help="budget for experiments that do not fix their own")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("presets:")
        for k, v in PRESETS.items():
            print("  %-11s %s" % (k, " ".join(v)))
        print("\nexperiments:")
        for k, v in EXPERIMENTS.items():
            print("  %-22s %s" % (k, v))
        return 0

    recs = load_all()
    if args.table:
        table(recs) if recs else print("還沒有任何結果")
        return 0

    if not os.path.exists(YAML):
        print("找不到 %s\n先跑: python ../Team13_Stage3/src/prepare_data.py "
              "--data_root <STUDENT_PUBLIC> --out %s" % (YAML, DS))
        return 2

    names = PRESETS.get(args.only, None)
    if names is None:
        names = args.only.split(",") if args.only else PRESETS["screen"]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print("unknown experiment(s):", ", ".join(unknown))
        print("try --list")
        return 2

    done = {r["name"] for r in recs}
    for name in names:
        if name in done and not args.force:
            print("[skip] %s 已有結果 (%.4f)"
                  % (name, next(r["map50_ours"] for r in recs
                                if r["name"] == name)), flush=True)
            continue
        print("\n" + "=" * 72)
        print("[run] %s  %s" % (name, EXPERIMENTS[name]))
        print("=" * 72, flush=True)
        try:
            rec = run_one(name, EXPERIMENTS[name], args.epochs, args.data_root)
        except RuntimeError as e:
            # an OOM on one setting must not kill the rest of the sweep
            print("[FAILED] %s  %s: %s" % (name, type(e).__name__,
                                           str(e)[:160]), flush=True)
            torch.cuda.empty_cache()
            continue
        recs = [r for r in recs if r["name"] != name] + [rec]
        save_all(recs)
        print("[done] %s  mAP@0.5 = %.4f  (%.0f 分鐘, %.0f s/epoch, "
              "peak %.0f MB)"
              % (name, rec["map50_ours"], rec["train_seconds"] / 60,
                 rec["sec_per_epoch"] or 0, rec["peak_vram_mb"]), flush=True)

    print()
    table(recs)
    print("\n結果檔: %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
