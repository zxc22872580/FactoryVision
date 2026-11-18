# TeamXX — FactoryVision Stage 2

Transfer Learning Challenge（鋼材表面瑕疵分類）

## 環境

- Python 3.11.9
- PyTorch 2.6.0+cu124 / torchvision 0.21.0+cu124
- 其餘套件見 `requirements.txt`

```
pip install -r requirements.txt
```

## Final Model

| 項目 | 內容 |
| --- | --- |
| Architecture | resnet18 |
| Strategy | l2sp |
| Pretrained source | torchvision ImageNet-1k (IMAGENET1K_V1) |
| Checkpoint | `stage2_model.pth` |
| 參數量 / 可訓練參數 | 11,179,590 / 11,179,590 |
| Validation Macro F1 | 1.0000 |

## TA 執行的固定指令

```
python stage2_inference.py --input_dir <DIR> --output_csv predictions.csv
```

同一支 script、同一份 checkpoint 連續執行 4 次（Standard / lighting / optical /
sensor_compression），輸出格式完全相同。程式**不會**讀取 `--input_dir` 的名稱，
也不會依 benchmark 類型切換 checkpoint、threshold、normalization 或 prediction rule。

## 模型輸入與 preprocessing

1. 以灰階讀圖後複製為 3 通道（`convert("L").convert("RGB")`）
2. `Resize((224, 224))`
3. `ToTensor()`
4. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`（ImageNet 統計值）

inference 使用 `model.eval()` + `torch.no_grad()`，不使用 TTA 或 Ensemble。
backbone 以 `pretrained=False` 建立後由 `stage2_model.pth` 載入權重，執行期完全離線。

## 目錄說明

```
TeamXX_Stage2/
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
