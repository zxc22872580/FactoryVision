"""Stage 1 Hidden Benchmark inference script.

Fixed command required by the TA:
    python stage1_inference.py --input_dir <HIDDEN_DIR> --output_csv predictions.csv

Behaviour
---------
* loads ``stage1_model.pth`` from the submission root (resolved relative to
  this file, never from the current working directory);
* applies exactly the same preprocessing as validation:
  grayscale -> Resize(128,128) -> ToTensor -> Normalize(mean,std);
* runs the model in ``eval()`` mode with ``torch.no_grad()``;
* no TTA, no ensembling, no randomness, no network access;
* writes ``image_id,prediction`` - one row per input image.
"""
import argparse
import csv
import os
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from common import CLASS_ORDER, INPUT_SIZE, IMG_EXT, NORM_MEAN, NORM_STD  # noqa: E402
from model import build_model  # noqa: E402

CHECKPOINT = os.path.join(HERE, "stage1_model.pth")


def build_transform(size):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    size = int(cfg.get("input_size", INPUT_SIZE))
    model = build_model(use_bn=cfg.get("bn", True),
                        dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    tf = build_transform(size)

    files = [f for f in sorted(os.listdir(args.input_dir))
             if f.lower().endswith(IMG_EXT)]
    if not files:
        raise SystemExit("no image found in " + args.input_dir)

    rows = []
    with torch.no_grad():
        for i in range(0, len(files), args.batch_size):
            chunk = files[i:i + args.batch_size]
            batch = torch.stack([
                tf(Image.open(os.path.join(args.input_dir, f)).convert("L"))
                for f in chunk]).to(device)
            pred = model(batch).argmax(1).cpu().tolist()
            rows.extend((os.path.splitext(f)[0], CLASS_ORDER[p])
                        for f, p in zip(chunk, pred))

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "prediction"])
        w.writerows(rows)
    print("wrote {} predictions -> {}".format(len(rows), args.output_csv))


if __name__ == "__main__":
    main()
