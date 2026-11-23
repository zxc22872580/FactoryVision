# Team13 — FactoryVision Stage 1

Baseline + Improved CNN（鋼材表面瑕疵分類）

## 環境

- Python 3.11.9
- PyTorch 2.6.0+cu124 / torchvision 0.21.0+cu124（CUDA 可用時自動使用 GPU，否則退回 CPU）
- 其餘套件見 `requirements.txt`

```
pip install -r requirements.txt
```

## Final Model

| 項目 | 內容 |
| --- | --- |
| Final Model | BaselineCNN(from scratch) + Data Augmentation + Batch Normalization |
| Checkpoint | `stage1_model.pth` |
| Pretrained source | 無（from scratch，未使用任何 pretrained weights） |
| Improvement Methods | Data Augmentation、Batch Normalization |
| 參數量 | 615,974 |
| Validation Macro F1 | 1.0000 |

## TA 執行的固定指令

```
python stage1_inference.py --input_dir <HIDDEN_DIR> --output_csv predictions.csv
```

輸出 `image_id,prediction` 兩個欄位，每張影像一列，`prediction` 為
crazing / inclusion / patches / pitted_surface / rolled_in_scale / scratches 之一。

## 模型輸入與 preprocessing

1. 以灰階（`convert("L")`）讀圖
2. `Resize((128, 128))`
3. `ToTensor()`
4. `Normalize(mean=[0.507], std=[0.2132])`

inference 使用 `model.eval()` + `torch.no_grad()`，不使用 TTA、Ensemble 或任何隨機性，
全程離線、不下載任何權重。checkpoint 以相對於 `stage1_inference.py` 的路徑載入。

## 目錄說明

```
Team13_Stage1/
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
