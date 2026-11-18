# -*- coding: utf-8 -*-
"""Create config.json + README.md and copy the final checkpoint for a stage."""
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEAM = "TeamXX"
CLASS_ORDER = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled_in_scale", "scratches"]

README1 = """# {team} — FactoryVision Stage 1

Baseline + Improved CNN（鋼材表面瑕疵分類）

## 環境

- Python 3.11.9
- PyTorch {torch} / torchvision {tv}（CUDA 可用時自動使用 GPU，否則退回 CPU）
- 其餘套件見 `requirements.txt`

```
pip install -r requirements.txt
```

## Final Model

| 項目 | 內容 |
| --- | --- |
| Final Model | {model_name} |
| Checkpoint | `stage1_model.pth` |
| Pretrained source | 無（from scratch，未使用任何 pretrained weights） |
| Improvement Methods | {improvements} |
| 參數量 | {params:,} |
| Validation Macro F1 | {f1} |

## TA 執行的固定指令

```
python stage1_inference.py --input_dir <HIDDEN_DIR> --output_csv predictions.csv
```

輸出 `image_id,prediction` 兩個欄位，每張影像一列，`prediction` 為
crazing / inclusion / patches / pitted_surface / rolled_in_scale / scratches 之一。

## 模型輸入與 preprocessing

1. 以灰階（`convert("L")`）讀圖
2. `Resize(({size}, {size}))`
3. `ToTensor()`
4. `Normalize(mean={mean}, std={std})`

inference 使用 `model.eval()` + `torch.no_grad()`，不使用 TTA、Ensemble 或任何隨機性，
全程離線、不下載任何權重。checkpoint 以相對於 `stage1_inference.py` 的路徑載入。

## 目錄說明

```
{team}_Stage1/
├─ stage1_report.pdf      完整實驗報告
├─ stage1_inference.py    Benchmark inference（唯一入口）
├─ stage1_model.pth       Final Improved CNN checkpoint
├─ config.json
├─ requirements.txt
├─ README.md
└─ src/
   ├─ common.py             常數、split 列舉、seed
   ├─ model.py              Baseline CNN 架構（use_bn 開關 = Improvement B）
   ├─ dataset.py            Dataset 與 transform（augment 開關 = Improvement A）
   ├─ train.py              訓練主程式（E1–E4 ablation 皆由此執行）
   ├─ evaluate.py           Metrics / Confusion Matrix / 延遲量測
   ├─ stress_eval.py        劣化條件下的次要評估（選模輔助）
   └─ traditional_baseline.py  PCA + SVM / PCA + KNN
```

## 重現實驗

```
cd src
python traditional_baseline.py --data_root <STUDENT_PUBLIC> --out_dir ../../work/outputs
python train.py --data_root <STUDENT_PUBLIC> --exp E1_baseline            --epochs 80
python train.py --data_root <STUDENT_PUBLIC> --exp E2_aug      --augment   --epochs 80
python train.py --data_root <STUDENT_PUBLIC> --exp E3_bn                --bn --epochs 80
python train.py --data_root <STUDENT_PUBLIC> --exp E4_aug_bn   --augment --bn --epochs 80
```

所有實驗固定 `random seed = 42`。
"""

README2 = """# {team} — FactoryVision Stage 2

Transfer Learning Challenge（鋼材表面瑕疵分類）

## 環境

- Python 3.11.9
- PyTorch {torch} / torchvision {tv}
- 其餘套件見 `requirements.txt`

```
pip install -r requirements.txt
```

## Final Model

| 項目 | 內容 |
| --- | --- |
| Architecture | {arch} |
| Strategy | {strategy} |
| Pretrained source | {pretrained} |
| Checkpoint | `stage2_model.pth` |
| 參數量 / 可訓練參數 | {params:,} / {trainable:,} |
| Validation Macro F1 | {f1} |

## TA 執行的固定指令

```
python stage2_inference.py --input_dir <DIR> --output_csv predictions.csv
```

同一支 script、同一份 checkpoint 連續執行 4 次（Standard / lighting / optical /
sensor_compression），輸出格式完全相同。程式**不會**讀取 `--input_dir` 的名稱，
也不會依 benchmark 類型切換 checkpoint、threshold、normalization 或 prediction rule。

## 模型輸入與 preprocessing

1. 以灰階讀圖後複製為 3 通道（`convert("L").convert("RGB")`）
2. `Resize(({size}, {size}))`
3. `ToTensor()`
4. `Normalize(mean={mean}, std={std})`（ImageNet 統計值）

inference 使用 `model.eval()` + `torch.no_grad()`，不使用 TTA 或 Ensemble。
backbone 以 `pretrained=False` 建立後由 `stage2_model.pth` 載入權重，執行期完全離線。

## 目錄說明

```
{team}_Stage2/
├─ stage2_report.pdf
├─ stage2_inference.py
├─ stage2_model.pth
├─ config.json
├─ requirements.txt
├─ README.md
└─ src/
   ├─ common.py     常數與 split 列舉
   ├─ model.py      三種 architecture + 四種 fine-tuning strategy（含 L2-SP）
   ├─ dataset.py    共用的 augmentation pipeline（對應三種 robustness）
   ├─ train.py      單組實驗的訓練主程式
   └─ evaluate.py   Metrics、延遲量測、內部 robustness 探測
```

## 重現實驗

```
cd src
python train.py --data_root <ROOT> --arch resnet18    --strategy feature_extraction --epochs 20 --batch_size 16 --amp
python train.py --data_root <ROOT> --arch resnet18    --strategy full_finetune      --epochs 20 --batch_size 16 --amp
python train.py --data_root <ROOT> --arch densenet121 --strategy feature_extraction --epochs 20 --batch_size 16 --amp
python train.py --data_root <ROOT> --arch densenet121 --strategy full_finetune      --epochs 20 --batch_size 16 --amp
```

所有實驗固定 `random seed = 42`。
"""

README3 = """# {team} — FactoryVision Stage 3

Object Detection（鋼材表面瑕疵定位）

## 環境

- Python 3.11.9
- PyTorch {torch} / torchvision {tv} / ultralytics {ul}
- 其餘套件見 `requirements.txt`

```
pip install -r requirements.txt
```

## Final Model

| 項目 | 內容 |
| --- | --- |
| Detector | {detector} |
| Backbone | {backbone} |
| Pretrained source | {pretrained} |
| Checkpoint | `stage3_model.pt` |
| Input size | {size}×{size} |
| Confidence threshold | {conf} |
| NMS IoU threshold | {nms} |
| 每張最多輸出框數 | {maxdet} |
| Validation mAP@0.5 | {map50} |

## TA 執行的固定指令

```
python stage3_detect.py --input_dir <HIDDEN_DIR> --output_json detections.json
```

輸出為 JSON array，每張輸入影像恰好一個物件；沒有偵測結果時 `detections` 為 `[]`。
`bbox` 為原始影像 pixel 座標的 `[xmin, ymin, xmax, ymax]`，已 clip 在影像範圍內。

## 門檻說明

Confidence threshold 刻意設為 {conf}（很低）：官方 mAP 以整條 PR 曲線積分，
先過濾低分框只會讓 recall 端被截斷、直接降低 mAP。
若要給產線作業員看，建議在下游 UI 以較高門檻（約 0.25–0.40）過濾。

## 目錄說明

```
{team}_Stage3/
├─ stage3_report.pdf
├─ stage3_detect.py
├─ stage3_model.pt
├─ config.json
├─ requirements.txt
├─ README.md
└─ src/
   ├─ common.py        常數、VOC 解析（含 rolled-in_scale 名稱正規化）
   ├─ model.py         detector 載入與參數統計
   ├─ prepare_data.py  由公開 split 建立 Ultralytics 資料夾結構
   ├─ train.py         訓練主程式
   ├─ evaluate.py      依題目定義實作的 mAP@0.5（VOC 2010+ all-points）
   └─ visualize.py     GT vs Prediction 視覺化、失敗案例、資料統計
```

## 重現實驗

```
cd src
python prepare_data.py --data_root <STUDENT_PUBLIC> --out <DS_DIR>
python train.py --data_yaml <DS_DIR>/factoryvision.yaml --epochs {epochs} --imgsz {size} --batch {batch}
```

所有實驗固定 `random seed = 42`。
"""


def versions():
    import torch
    import torchvision
    v = {"torch": torch.__version__, "tv": torchvision.__version__}
    try:
        import ultralytics
        v["ul"] = ultralytics.__version__
    except ImportError:
        v["ul"] = "8.4.215"
    return v


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("->", path)


def stage1(args):
    import torch
    dst = os.path.join(ROOT, TEAM + "_Stage1")
    summary = json.load(open(args.summary, encoding="utf-8"))
    cfg_run = summary["config"]
    shutil.copyfile(args.checkpoint, os.path.join(dst, "stage1_model.pth"))
    blob = torch.load(os.path.join(dst, "stage1_model.pth"),
                      map_location="cpu", weights_only=False)
    improvements = []
    if cfg_run["augment"]:
        improvements.append("Data Augmentation")
    if cfg_run["bn"]:
        improvements.append("Batch Normalization")
    cfg = {
        "team_id": TEAM, "stage": 1,
        "model_name": "BaselineCNN(from scratch) + " + " + ".join(improvements),
        "input_size": [cfg_run["input_size"], cfg_run["input_size"]],
        "class_order": CLASS_ORDER,
        "normalization_mean": cfg_run["norm_mean"],
        "normalization_std": cfg_run["norm_std"],
        "seed": cfg_run["seed"],
        "improvement_methods": improvements,
        "checkpoint_file": "stage1_model.pth",
        "pretrained_source": None,
        "improvement_details": {
            "Data Augmentation": cfg_run.get("aug_config"),
            "Batch Normalization": {
                "where": "after every Conv2d, before ReLU",
                "num_layers": 7, "momentum": 0.1, "eps": 1e-5},
        },
        "training": {k: cfg_run[k] for k in
                     ("lr", "batch_size", "epochs", "weight_decay", "dropout",
                      "merge_train_val")},
        "model_selection": {
            "selected_experiment": args.selected_exp,
            "selection_run_validation_macro_f1": args.selection_f1,
            "note": ("The submitted checkpoint was retrained on train+val "
                     "after model selection, so the validation split is part "
                     "of its training data and its validation score is no "
                     "longer an independent estimate."),
        },
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    print("-> config.json")
    v = versions()
    write(os.path.join(dst, "README.md"), README1.format(
        team=TEAM, torch=v["torch"], tv=v["tv"],
        model_name=cfg["model_name"], improvements="、".join(improvements),
        params=summary["params"], f1="{:.4f}".format(args.selection_f1),
        size=cfg_run["input_size"], mean=cfg_run["norm_mean"],
        std=cfg_run["norm_std"]))
    assert blob["config"]["bn"] == cfg_run["bn"]


def stage2(args):
    dst = os.path.join(ROOT, TEAM + "_Stage2")
    summary = json.load(open(args.summary, encoding="utf-8"))
    c = summary["config"]
    shutil.copyfile(args.checkpoint, os.path.join(dst, "stage2_model.pth"))
    cfg = {
        "team_id": TEAM, "stage": 2,
        "architecture": c["architecture"], "strategy": c["strategy"],
        "pretrained_source": c["pretrained_source"],
        "input_size": [c["input_size"], c["input_size"]],
        "class_order": CLASS_ORDER,
        "normalization_mean": c["norm_mean"],
        "normalization_std": c["norm_std"],
        "seed": c["seed"], "checkpoint_file": "stage2_model.pth",
        "frozen_layers": c["frozen_layers"],
        "trainable_layers": c["trainable_layers"],
        "training": {k: c[k] for k in ("lr", "weight_decay", "batch_size",
                                       "epochs", "label_smoothing",
                                       "merge_train_val")},
        "train_time_augmentation": c.get("robust_augmentation"),
        "model_selection": {
            "selected_experiment": args.selected_exp,
            "selection_run_validation_macro_f1": args.selection_f1,
            "note": ("The submitted checkpoint was retrained on train+val "
                     "after model selection; its validation score is "
                     "therefore not an independent estimate."),
        },
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    print("-> config.json")
    v = versions()
    write(os.path.join(dst, "README.md"), README2.format(
        team=TEAM, torch=v["torch"], tv=v["tv"], arch=c["architecture"],
        strategy=c["strategy"], pretrained=c["pretrained_source"],
        params=summary["params"], trainable=summary["trainable_params"],
        f1="{:.4f}".format(args.selection_f1),
        size=c["input_size"], mean=c["norm_mean"], std=c["norm_std"]))


def stage3(args):
    dst = os.path.join(ROOT, TEAM + "_Stage3")
    res = json.load(open(args.summary, encoding="utf-8"))
    tcfg = json.load(open(args.train_config, encoding="utf-8"))
    shutil.copyfile(args.checkpoint, os.path.join(dst, "stage3_model.pt"))
    cfg = {
        "team_id": TEAM, "stage": 3,
        "detector": tcfg["detector"], "backbone": tcfg["backbone"],
        "pretrained_source": tcfg["pretrained_source"],
        "input_size": [res["imgsz"], res["imgsz"]],
        "class_order": CLASS_ORDER,
        "confidence_threshold": res["conf_threshold"],
        "nms_iou_threshold": res["nms_iou_threshold"],
        "bbox_format": "xyxy_pixel",
        "max_detections_per_image": res["max_det"],
        "seed": tcfg.get("seed", 42),
        "checkpoint_file": "stage3_model.pt",
        "training": {k: tcfg.get(k) for k in
                     ("epochs", "batch", "lr0", "lrf", "optimizer",
                      "patience", "imgsz")},
        "augmentation": tcfg.get("augmentation"),
        "validation_map50": res["validation"]["map50"],
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    print("-> config.json")
    v = versions()
    write(os.path.join(dst, "README.md"), README3.format(
        team=TEAM, torch=v["torch"], tv=v["tv"], ul=v["ul"],
        detector=tcfg["detector"], backbone=tcfg["backbone"],
        pretrained=tcfg["pretrained_source"], size=res["imgsz"],
        conf=res["conf_threshold"], nms=res["nms_iou_threshold"],
        maxdet=res["max_det"],
        map50="{:.4f}".format(res["validation"]["map50"]),
        epochs=tcfg.get("epochs"), batch=tcfg.get("batch")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--train_config", default=None)
    ap.add_argument("--selected_exp", default=None,
                    help="name of the model-selection run this checkpoint "
                         "reproduces")
    ap.add_argument("--selection_f1", type=float, default=None,
                    help="that run's independent validation Macro F1")
    a = ap.parse_args()
    {1: stage1, 2: stage2, 3: stage3}[a.stage](a)
