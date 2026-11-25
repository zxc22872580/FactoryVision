# -*- coding: utf-8 -*-
"""Find a (model, batch, imgsz) setting that fits this 4 GB GPU.

Two epochs each; epoch-2 minus epoch-1 wall time is the steady-state cost.
A peak above ~4000 MB means the allocator is spilling into shared system
memory over WDDM, which roughly halves throughput, so a setting that reports
less VRAM can easily be the faster one.

Must stay under ``if __name__ == "__main__"`` -- Windows uses spawn, and the
Ultralytics dataloader starts worker processes.
"""
import csv
import os
import time

import torch
from ultralytics import YOLO

W = os.path.dirname(os.path.abspath(__file__))
YAML = os.path.join(W, "det_dataset", "factoryvision.yaml")

GRID = [("yolov8s", 8, 640),     # the submitted setting, as a control
        ("yolo11s", 4, 640),
        ("yolo11s", 8, 512),
        ("yolo11s", 16, 416),
        ("yolov9s", 8, 512),
        ("yolov10s", 8, 512),
        ("yolo12s", 8, 512)]


def main():
    print("%-20s %10s %10s %9s" % ("setting", "s/epoch", "peak MB", "status"))
    for mdl, batch, imgsz in GRID:
        name = "%s_b%d_%d" % (mdl, batch, imgsz)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            YOLO(os.path.join(W, mdl + ".pt")).train(
                data=YAML, epochs=2, batch=batch, imgsz=imgsz, cache=True,
                workers=4, seed=42, deterministic=True, patience=100,
                project=os.path.join(W, "stage3_probe"), name=name,
                exist_ok=True, verbose=False, plots=False, val=False)
            rows = list(csv.DictReader(
                open(os.path.join(W, "stage3_probe", name, "results.csv"))))
            t = [float(r["time"]) for r in rows]
            sec = (t[1] - t[0]) if len(t) > 1 else (time.time() - t0)
            print("%-20s %10.1f %10.0f %9s"
                  % (name, sec, torch.cuda.max_memory_reserved() / 1e6, "ok"),
                  flush=True)
        except Exception as e:
            print("%-20s %10s %10s %9s  %s"
                  % (name, "-", "-", "FAILED",
                     type(e).__name__ + ": " + str(e)[:80]), flush=True)


if __name__ == "__main__":
    main()
