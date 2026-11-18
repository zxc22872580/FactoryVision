"""Stage 3 - fine-tune the YOLOv8s detector on the FactoryVision split.

    python prepare_data.py --data_root <STUDENT_PUBLIC> --out <DS>
    python train.py --data_yaml <DS>/factoryvision.yaml --epochs 150

The TA-provided train/validation split is used as-is; ``val`` in the yaml is
the public validation split, so early stopping / model selection only ever
sees public data.
"""
import argparse
import json
import os

from common import INPUT_SIZE, SEED, set_seed
from model import BACKBONE, DEFAULT_PRETRAINED, PRETRAINED_SOURCE, \
    build_for_training, model_info


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_yaml", required=True)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=INPUT_SIZE)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="yolov8s_640")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--cache", action="store_true")
    # augmentation (recorded in the report)
    p.add_argument("--degrees", type=float, default=10.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--flipud", type=float, default=0.5)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--close_mosaic", type=int, default=20)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--hsv_h", type=float, default=0.0)
    p.add_argument("--hsv_s", type=float, default=0.0)
    p.add_argument("--hsv_v", type=float, default=0.4)
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    model = build_for_training(args.pretrained)
    info = model_info(model)
    print("detector=YOLOv8s backbone={} params={:,}".format(BACKBONE,
                                                            info["params"]))

    results = model.train(
        data=args.data_yaml, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, patience=args.patience, lr0=args.lr0, lrf=args.lrf,
        optimizer=args.optimizer, project=args.project, name=args.name,
        device=args.device, workers=args.workers, seed=args.seed,
        deterministic=True, cache=args.cache, exist_ok=True, plots=True,
        degrees=args.degrees, translate=args.translate, scale=args.scale,
        fliplr=args.fliplr, flipud=args.flipud, mosaic=args.mosaic,
        close_mosaic=args.close_mosaic, mixup=args.mixup,
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
    )

    save_dir = str(results.save_dir)
    cfg = {
        "detector": "YOLOv8s", "backbone": BACKBONE,
        "pretrained_source": PRETRAINED_SOURCE,
        "imgsz": args.imgsz, "epochs": args.epochs, "batch": args.batch,
        "patience": args.patience, "lr0": args.lr0, "lrf": args.lrf,
        "optimizer": args.optimizer, "seed": args.seed,
        "augmentation": {"degrees": args.degrees, "translate": args.translate,
                         "scale": args.scale, "fliplr": args.fliplr,
                         "flipud": args.flipud, "mosaic": args.mosaic,
                         "close_mosaic": args.close_mosaic,
                         "mixup": args.mixup, "hsv_h": args.hsv_h,
                         "hsv_s": args.hsv_s, "hsv_v": args.hsv_v},
        "params": info["params"], "save_dir": save_dir,
        "best_weights": os.path.join(save_dir, "weights", "best.pt"),
    }
    with open(os.path.join(save_dir, "train_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("training finished ->", cfg["best_weights"])


if __name__ == "__main__":
    main()
