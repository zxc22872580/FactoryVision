"""Stage 2 Standard / Robustness Benchmark inference script.

Fixed command required by the TA (run 4 times, on 4 different folders):
    python stage2_inference.py --input_dir <DIR> --output_csv predictions.csv

The script is intentionally *blind* to which benchmark it is running on:

* one single checkpoint (``stage2_model.pth``) resolved relative to this file;
* one single preprocessing pipeline
  (grayscale -> RGB -> Resize(224,224) -> ToTensor -> ImageNet Normalize);
* no branching on ``--input_dir`` name, no threshold switching, no TTA,
  no ensembling, no randomness, no network access.
"""
import argparse
import csv
import os
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from common import CLASS_ORDER, IMG_EXT, INPUT_SIZE, NORM_MEAN, NORM_STD  # noqa: E402
from model import build_backbone  # noqa: E402

CHECKPOINT = os.path.join(HERE, "stage2_model.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    size = int(cfg.get("input_size", INPUT_SIZE))
    # pretrained=False: weights come from the bundled checkpoint, never from
    # the network - the script must run fully offline.
    model, _ = build_backbone(cfg["architecture"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.get("norm_mean", NORM_MEAN),
                             cfg.get("norm_std", NORM_STD)),
    ])

    files = [f for f in sorted(os.listdir(args.input_dir))
             if f.lower().endswith(IMG_EXT)]
    if not files:
        raise SystemExit("no image found in " + args.input_dir)

    rows = []
    with torch.no_grad():
        for i in range(0, len(files), args.batch_size):
            chunk = files[i:i + args.batch_size]
            batch = torch.stack([
                tf(Image.open(os.path.join(args.input_dir, f))
                   .convert("L").convert("RGB")) for f in chunk]).to(device)
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
