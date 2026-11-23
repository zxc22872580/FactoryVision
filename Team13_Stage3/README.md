# Team13 — FactoryVision Stage 3

Object Detection（鋼材表面瑕疵定位）

## 環境

- Python 3.11.9
- PyTorch 2.6.0+cu124 / torchvision 0.21.0+cu124 / ultralytics 8.4.215
- 其餘套件見 `requirements.txt`

```
pip install -r requirements.txt
```

## Final Model

| 項目 | 內容 |
| --- | --- |
| Detector | YOLOv8s |
| Backbone | CSPDarknet (YOLOv8s) |
| Pretrained source | Ultralytics YOLOv8s COCO-pretrained weights (yolov8s.pt) |
| Checkpoint | `stage3_model.pt` |
| Input size | 640×640 |
| Confidence threshold | 0.001 |
| NMS IoU threshold | 0.6 |
| 每張最多輸出框數 | 100 |
| Validation mAP@0.5 | 0.7419 |

## TA 執行的固定指令

```
python stage3_detect.py --input_dir <HIDDEN_DIR> --output_json detections.json
```

輸出為 JSON array，每張輸入影像恰好一個物件；沒有偵測結果時 `detections` 為 `[]`。
`bbox` 為原始影像 pixel 座標的 `[xmin, ymin, xmax, ymax]`，已 clip 在影像範圍內。

## 門檻說明

Confidence threshold 刻意設為 0.001（很低）：官方 mAP 以整條 PR 曲線積分，
先過濾低分框只會讓 recall 端被截斷、直接降低 mAP。
若要給產線作業員看，建議在下游 UI 以較高門檻（約 0.25–0.40）過濾。

## 目錄說明

```
Team13_Stage3/
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
python train.py --data_yaml <DS_DIR>/factoryvision.yaml --epochs 100 --imgsz 640 --batch 8
```

所有實驗固定 `random seed = 42`。
