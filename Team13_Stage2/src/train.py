"""Stage 2 - training entry point for the 2 x 2 transfer-learning comparison.

Every run uses the identical data split, augmentation pipeline, optimizer
family, epoch budget, early-stopping rule and evaluation code; only
``--arch`` and ``--strategy`` change.

Example (the four required experiments):
    python train.py --data_root <ROOT> --arch resnet18    --strategy feature_extraction
    python train.py --data_root <ROOT> --arch resnet18    --strategy full_finetune
    python train.py --data_root <ROOT> --arch densenet121 --strategy feature_extraction
    python train.py --data_root <ROOT> --arch densenet121 --strategy full_finetune
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (CLASS_ORDER, INPUT_SIZE, NORM_MEAN, NORM_STD, SEED,
                    list_split, set_seed)
from dataset import AUG_CONFIG, build_datasets, eval_transform
from evaluate import (compute_metrics, measure_latency, predict,
                      robustness_report)
from model import ARCHITECTURES, L2SP, STRATEGIES, build, count_parameters


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--arch", required=True, choices=ARCHITECTURES)
    p.add_argument("--strategy", required=True, choices=STRATEGIES)
    p.add_argument("--last_k", type=int, default=2)
    p.add_argument("--l2sp_alpha", type=float, default=0.01)
    p.add_argument("--l2sp_beta", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-3 for frozen backbone, 1e-4 otherwise")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--input_size", type=int, default=INPUT_SIZE)
    p.add_argument("--no_robust_aug", action="store_true")
    p.add_argument("--merge_train_val", action="store_true")
    p.add_argument("--select_last", action="store_true",
                   help="keep the last epoch instead of the best-by-val one; "
                        "used for the train+val retraining, where validation "
                        "is part of the training data")
    p.add_argument("--amp", action="store_true", help="mixed precision")
    p.add_argument("--tag", default=None)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip_robustness", action="store_true")
    return p.parse_args()


def run_epoch(model, loader, device, criterion, optimizer=None, reg=None,
              scaler=None):
    train = optimizer is not None
    model.train(train)
    tot, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                    loss = criterion(out, y)
                if reg is not None:
                    loss = loss + reg(model)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                if train:
                    if reg is not None:
                        loss = loss + reg(model)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
        tot += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return tot / n, correct / n


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or "{}_{}".format(args.arch, args.strategy)
    if args.lr is None:
        args.lr = 1e-3 if args.strategy == "feature_extraction" else 1e-4

    train_ds, val_ds = build_datasets(args.data_root, size=args.input_size,
                                      robust=not args.no_robust_aug,
                                      merge_train_val=args.merge_train_val)
    # persistent_workers avoids re-spawning (and re-importing torch in) every
    # worker at every epoch, which is very expensive on Windows.
    dl_kw = dict(num_workers=args.num_workers)
    if args.num_workers > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=4,
                     pin_memory=True)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          **dl_kw)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        **dl_kw)

    model, info = build(args.arch, args.strategy, pretrained=True,
                        last_k=args.last_k)
    model.to(device)
    total, trainable = count_parameters(model)

    reg = None
    if args.strategy == "l2sp":
        reg = L2SP(model, info["head_attr"], args.l2sp_alpha,
                   args.l2sp_beta).to(device)
        wd = 0.0
    else:
        wd = args.weight_decay

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and
                                              device.type == "cuda") else None

    print("[{}] device={} lr={} bs={} params={:,} trainable={:,} "
          "({:.1f}%) train_n={}".format(tag, device, args.lr, args.batch_size,
                                        total, trainable,
                                        100.0 * trainable / total,
                                        len(train_ds)))
    print("   frozen  :", info["frozen_layers"])
    print("   trainable:", info["trainable_layers"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "val_macro_f1": []}
    best_f1, best_state, best_epoch, bad = -1.0, None, -1, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_ld, device, criterion, optimizer, reg,
                           scaler)
        vl, va = run_epoch(model, val_ld, device, criterion)
        pred, true = predict(model, val_ld, device)
        vf1 = compute_metrics(true, pred)["macro_f1"]
        scheduler.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["val_macro_f1"].append(vf1)
        flag = ""
        if args.select_last:
            best_f1, best_epoch, bad = vf1, ep, 0
            best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in
                 model.state_dict().items()})
        elif vf1 > best_f1:
            best_f1, best_epoch, bad = vf1, ep, 0
            best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in
                 model.state_dict().items()})
            flag = " *"
        else:
            bad += 1
        print("  ep{:3d} tl={:.4f} ta={:.4f} | vl={:.4f} va={:.4f} "
              "vF1={:.4f}{}".format(ep, tl, ta, vl, va, vf1, flag))
        if bad >= args.patience:
            print("  early stop at epoch {}".format(ep))
            break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    model.to(device)
    pred, true = predict(model, val_ld, device)
    metrics = compute_metrics(true, pred)

    ckpt_path = os.path.join(args.out_dir, tag + ".pth")
    config = {
        "tag": tag, "architecture": args.arch, "strategy": args.strategy,
        "pretrained_source": "torchvision ImageNet-1k (IMAGENET1K_V1)",
        "last_k": info["last_k"], "frozen_layers": info["frozen_layers"],
        "trainable_layers": info["trainable_layers"],
        "l2sp_alpha": args.l2sp_alpha if args.strategy == "l2sp" else None,
        "l2sp_beta": args.l2sp_beta if args.strategy == "l2sp" else None,
        "lr": args.lr, "weight_decay": wd, "batch_size": args.batch_size,
        "epochs": args.epochs, "label_smoothing": args.label_smoothing,
        "input_size": args.input_size, "seed": args.seed,
        "class_order": CLASS_ORDER, "norm_mean": NORM_MEAN,
        "norm_std": NORM_STD,
        "robust_augmentation": None if args.no_robust_aug else AUG_CONFIG,
        "merge_train_val": args.merge_train_val,
        "select_last": args.select_last,
    }
    torch.save({"state_dict": best_state, "config": config}, ckpt_path)

    summary = {
        "tag": tag, "config": config, "history": history,
        "best_epoch": best_epoch, "epochs_run": len(history["train_loss"]),
        "train_seconds": train_time, "params": total,
        "trainable_params": trainable,
        "trainable_ratio": trainable / total,
        "checkpoint_mb": os.path.getsize(ckpt_path) / 1024 ** 2,
        "latency_ms_bs1": measure_latency(model, device, args.input_size),
        "val": metrics,
    }

    if not args.skip_robustness:
        vp, vy = list_split(os.path.join(args.data_root, "classification",
                                         "validation"))
        summary["robustness"] = robustness_report(
            model, vp, vy, eval_transform(args.input_size), device,
            args.batch_size)
        print("  robust macro F1 (internal probe) = {:.4f}".format(
            summary["robustness"]["robust_macro_f1"]))

    with open(os.path.join(args.out_dir, tag + "_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[{}] best ep {} | val acc={:.4f} macroF1={:.4f} | {:.0f}s | "
          "ckpt {:.1f} MB".format(tag, best_epoch, metrics["accuracy"],
                                  metrics["macro_f1"], train_time,
                                  summary["checkpoint_mb"]))
    print("  confusion matrix:\n", np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()
