"""Stage 3 Hidden Benchmark detection script.

Fixed command required by the TA:
    python stage3_detect.py --input_dir <HIDDEN_DIR> --output_json detections.json

Guarantees
----------
* loads ``stage3_model.pt`` from the submission root (path resolved relative
  to this file, never from the current working directory);
* fully offline - Ultralytics is given a local .pt file, nothing is fetched;
* one JSON object per input image, ``detections: []`` when nothing is found;
* bbox = [xmin, ymin, xmax, ymax] in original-image pixel coordinates,
  clipped to the image, at most 100 boxes per image;
* confidence threshold 0.01 and NMS IoU 0.6 are fixed constants, identical
  for every input folder.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from common import (CLASS_ORDER, CONF_THRESHOLD, IMG_EXT, INPUT_SIZE,  # noqa: E402
                    MAX_DET, NMS_IOU_THRESHOLD)
from model import load_detector  # noqa: E402

CHECKPOINT = os.path.join(HERE, "stage3_model.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    import torch
    # fall back to CPU whenever no usable GPU is visible, so the benchmark
    # runs on a CPU-only machine as well
    try:
        use_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:                                       # noqa: BLE001
        use_gpu = False
    device = 0 if use_gpu else "cpu"

    model = load_detector(CHECKPOINT)
    files = [f for f in sorted(os.listdir(args.input_dir))
             if f.lower().endswith(IMG_EXT)]
    if not files:
        raise SystemExit("no image found in " + args.input_dir)

    out = []
    for i in range(0, len(files), args.batch_size):
        chunk = files[i:i + args.batch_size]
        paths = [os.path.join(args.input_dir, f) for f in chunk]
        results = model.predict(paths, imgsz=INPUT_SIZE, conf=CONF_THRESHOLD,
                                iou=NMS_IOU_THRESHOLD, max_det=MAX_DET,
                                device=device, verbose=False, augment=False)
        for fname, res in zip(chunk, results):
            h, w = res.orig_shape
            dets = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                order = conf.argsort()[::-1][:MAX_DET]
                for j in order:
                    x1 = float(min(max(xyxy[j][0], 0.0), w - 1.0))
                    y1 = float(min(max(xyxy[j][1], 0.0), h - 1.0))
                    x2 = float(min(max(xyxy[j][2], 0.0), float(w)))
                    y2 = float(min(max(xyxy[j][3], 0.0), float(h)))
                    if x2 <= x1 or y2 <= y1:     # degenerate box -> drop
                        continue
                    c = int(cls[j])
                    if not 0 <= c < len(CLASS_ORDER):
                        continue
                    dets.append({
                        "class": CLASS_ORDER[c],
                        "confidence": float(min(max(conf[j], 0.0), 1.0)),
                        "bbox": [round(x1, 2), round(y1, 2),
                                 round(x2, 2), round(y2, 2)],
                    })
            out.append({"image_id": os.path.splitext(fname)[0],
                        "detections": dets})

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("wrote {} images / {} detections -> {}".format(
        len(out), sum(len(r["detections"]) for r in out), args.output_json))


if __name__ == "__main__":
    main()
