"""Stage 1 - training entry point for the Baseline CNN and its ablations.

Improvement Method A : Data Augmentation   (--augment)
Improvement Method B : Batch Normalization (--bn)

The four required ablation experiments are therefore:
    E1 baseline       : (no flag)
    E2 baseline + A   : --augment
    E3 baseline + B   : --bn
    E4 baseline + A+B : --augment --bn

Early stopping on validation Macro-F1 is used in *every* experiment as a shared
training protocol, so that the ablation isolates the effect of A and B only.

Example:
    python train.py --data_root <STUDENT_PUBLIC> --exp E4 --augment --bn
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

from common import CLASS_ORDER, INPUT_SIZE, NORM_MEAN, NORM_STD, SEED, set_seed
from dataset import AUG_CONFIG, SteelDataset, build_datasets
from evaluate import compute_metrics, measure_latency, predict
from model import build_model, count_parameters


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--exp", default="E1", help="experiment name/tag")
    p.add_argument("--augment", action="store_true",
                   help="Improvement Method A: data augmentation")
    p.add_argument("--bn", action="store_true",
                   help="Improvement Method B: batch normalization")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20,
                   help="early-stopping patience on val macro F1")
    p.add_argument("--input_size", type=int, default=INPUT_SIZE)
    p.add_argument("--merge_train_val", action="store_true",
                   help="final retraining on train+val after model selection")
    p.add_argument("--select_last", action="store_true",
                   help="keep the final epoch instead of the best-by-val one; "
                        "used for the train+val retraining, where the "
                        "validation split is part of the training data and "
                        "can no longer be used for model selection")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return tot_loss / n, correct / n


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, val_ds = build_datasets(args.data_root, augment=args.augment,
                                      size=args.input_size,
                                      merge_train_val=args.merge_train_val)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = build_model(use_bn=args.bn, dropout=args.dropout).to(device)
    total, trainable = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    print("[{}] device={} augment={} bn={} dropout={} wd={} params={:,} "
          "train_n={} val_n={}".format(args.exp, device, args.augment, args.bn,
                                       args.dropout, args.weight_decay, total,
                                       len(train_ds), len(val_ds)))

    history = {"train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "val_macro_f1": [], "lr": []}
    best_f1, best_state, best_epoch, bad = -1.0, None, -1, 0
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_ld, device, criterion, optimizer)
        vl, va = run_epoch(model, val_ld, device, criterion, None)
        pred, true, _ = predict(model, val_ld, device)
        vf1 = compute_metrics(true, pred)["macro_f1"]
        scheduler.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["val_macro_f1"].append(vf1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        flag = ""
        if args.select_last:
            best_f1, best_epoch, bad = vf1, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        elif vf1 > best_f1:
            best_f1, best_epoch, bad = vf1, ep, 0
            best_state = copy.deepcopy(model.state_dict())
            flag = " *"
        else:
            bad += 1
        if ep % 5 == 0 or flag or ep == 1:
            print("  ep{:3d} tl={:.4f} ta={:.4f} | vl={:.4f} va={:.4f} "
                  "vF1={:.4f}{}".format(ep, tl, ta, vl, va, vf1, flag))
        if bad >= args.patience:
            print("  early stop at epoch {} (no improvement for {} epochs)"
                  .format(ep, args.patience))
            break
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    pred, true, _ = predict(model, val_ld, device)
    metrics = compute_metrics(true, pred)
    # training-set metrics under the *evaluation* transform (no augmentation),
    # used for the overfitting / generalization-gap analysis
    tr_eval_ld = DataLoader(
        SteelDataset(train_ds.paths, train_ds.labels, val_ds.transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)
    trp, trt, _ = predict(model, tr_eval_ld, device)
    train_metrics = compute_metrics(trt, trp)

    ckpt_path = os.path.join(args.out_dir, args.exp + ".pth")
    config = {
        "exp": args.exp, "augment": args.augment, "bn": args.bn,
        "dropout": args.dropout, "weight_decay": args.weight_decay,
        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "input_size": args.input_size, "seed": args.seed,
        "merge_train_val": args.merge_train_val,
        "select_last": args.select_last,
        "class_order": CLASS_ORDER, "norm_mean": NORM_MEAN,
        "norm_std": NORM_STD,
        "aug_config": AUG_CONFIG if args.augment else None,
    }
    torch.save({"state_dict": best_state, "config": config}, ckpt_path)

    summary = {
        "exp": args.exp, "config": config, "history": history,
        "best_epoch": best_epoch, "epochs_run": len(history["train_loss"]),
        "train_seconds": train_time,
        "params": total, "trainable_params": trainable,
        "checkpoint_mb": os.path.getsize(ckpt_path) / 1024 ** 2,
        "latency_ms_bs1": measure_latency(model, device, args.input_size),
        "val": metrics, "train": train_metrics,
        "train_acc_at_best": train_metrics["accuracy"],
        "generalization_gap_acc": (train_metrics["accuracy"]
                                   - metrics["accuracy"]),
    }
    with open(os.path.join(args.out_dir, args.exp + "_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[{}] best epoch {} | val acc={:.4f} macroF1={:.4f} | train acc="
          "{:.4f} | {:.0f}s".format(args.exp, best_epoch, metrics["accuracy"],
                                    metrics["macro_f1"],
                                    train_metrics["accuracy"], train_time))
    print("  per-class F1:", {c: round(v, 3) for c, v in
                              zip(CLASS_ORDER, metrics["per_class_f1"])})
    print("  confusion matrix:\n", np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()
