# Stage 3 調參／換模型：怎麼跑、要回報什麼

目標：把目前 **mAP@0.5 = 0.7419** 的 YOLOv8s 往上推。
兩條路都準備好了，**挑一條跑就好**，不用兩邊都做。

---

## 先搞清楚要解什麼問題

低分集中在兩個類別，不是整體都差：

| 類別 | AP@0.5 | recall@0.25 | 狀況 |
|---|---|---|---|
| **crazing** | **0.456** | 0.373 | 102 個 GT 漏掉 64 個 |
| **rolled_in_scale** | **0.581** | 0.548 | 93 個漏掉 42 個 |
| inclusion | 0.781 | 0.752 | |
| pitted_surface | 0.849 | 0.788 | |
| scratches | 0.858 | 0.899 | |
| patches | 0.928 | 0.908 | |

光把這兩類拉到 0.70，mAP 就會變 ~0.80。

而且 **原本那次訓練還沒收斂就被停掉了**。`work/stage3_runs/yolov8s_640/results.csv`：

```
epoch  81   mAP50 = 0.7294   mAP50-95 = 0.3824
epoch  91   mAP50 = 0.7277   mAP50-95 = 0.4020
epoch 100   mAP50 = 0.7413   mAP50-95 = 0.4188   <- 最後一個 epoch 還在爬
best   ep 95 mAP50 = 0.7531
```

`close_mosaic=20` 在第 80 epoch 關掉 mosaic 之後還在持續進步，到第 100 epoch 都沒平掉。
**所以「訓練久一點」是最便宜、最可能有效的一招**，比換模型優先。

---

## 規則確認（可以換模型、可以重訓）

| 規格書 | 內容 | 結論 |
|---|---|---|
| p.226 | 「可選 Faster R-CNN、SSD、YOLO、DETR 或經教師/TA 同意的等價模型」 | YOLO 全系列都在允許範圍 |
| p.59 | 「禁止使用已在網路上 fine-tune 完成的 checkpoint」 | COCO 預訓練權重沒問題 |
| p.434 | 「Final inference 不得依賴 runtime 下載 pretrained weights」 | 訓練時下載可以，**繳交包裡必須附權重** |
| p.271 | 「每個 Stage 僅允許 1 個 Benchmark checkpoint」 | 最後只能挑一顆進 zip |
| p.485 | 「不得以期末重新訓練的模型替換正式成績」 | TA 公告三個 stage 統一最後繳交，**還沒評分過，所以現在重訓就是正式成績** |

---

## 路線 A：Colab（建議）

T4 有 16 GB，batch 可以開 16 不溢出，而且有 tensor core 讓 AMP 真的生效。
**一個 s 級模型 100 epochs 約 35–40 分鐘**，整組五個約 3 小時。

### 步驟

1. 把 `FactoryVision_Colab.ipynb` 上傳到 Colab
2. 執行階段 → 變更執行階段類型 → **T4 GPU**
3. 把 `STUDENT_PUBLIC` 整個資料夾放到 Drive 的 `MyDrive/FactoryVision/`（103 MB，一次就好）
4. 依序執行：

   ```
   §1 全部  →  §2.1  →  §3.1 §3.2  →  §6.1  →  §6.4  →  §6.7a  →  §6.7b  →  §6.7c
   ```

   **§6.2、§6.3、§6.5、§6.6 不用跑**（那是單一模型的訓練與視覺化，sweep 不需要）。

5. `§6.7a` 選 preset：

   | preset | 內容 | T4 估計時間 |
   |---|---|---|
   | `screen` | v8s / 11s / v9s / v10s / 12s，各 100 ep @640 | ~3 h |
   | `long` | v8s 和 11s 各 250 ep | ~3 h |
   | `resolution` | 11s 在 416 / 512 / 640 / 800 | ~2 h |
   | `all` | 全部 13 組 | 太久，不建議一次 |

   **先跑 `screen`**，知道哪個架構好，再用那個架構跑 `long`。

6. `§6.7c` 最後會印一段 JSON → **整段複製貼回對話**

### Colab 的三個坑

- **閒置 90 分鐘會斷線**，分頁不要關、電腦不要睡
- 斷線後**重跑 `§6.7b` 就會接續**，已完成的實驗會自動跳過（結果存在 Drive）
- 免費版配額用完會被降成 CPU，那就完全跑不動。開始前先確認 §1.1 印出 T4

### 跑完第一個就看一下 s/epoch

`§6.7b` 每跑完一個會印實際的 `s/epoch`。我估 T4 約 20–25 s/epoch（batch 16 @640）。
如果印出來超過 40，可能是被配到比較慢的卡，那就把 preset 縮小。

---

## 路線 B：本機（GTX 1650 Ti 4 GB）

慢很多，但不用上傳、不會斷線。**實測速度**（`work/probe_speed.py`，2 epoch 取穩定值）：

```
setting              s/epoch   peak MB
yolov8s  b8  @640      99.6      3970    <- 繳交版的設定，已經貼到 4 GB 天花板
yolo11s  b4  @640     105.1      2655
yolo11s  b8  @512      73.0      3381    <- 比 v8s@640 還快
yolo11s  b16 @416      56.2      3817
yolov9s  b8  @512      80.3      3456
yolov10s b8  @512      79.5      3460
yolo12s  b8  @512      85.0      3928
```

> 峰值接近 4000 MB 就會溢出到共享記憶體（WDDM），速度直接砍半。
> yolo11s 在 b8@640 要 **224 s/epoch**，降到 512 之後只要 73 s —— 所以本機要用 512。
> 原圖只有 200×200，640 已經是 3.2 倍放大，降到 512 不一定會掉精度。

### 指令

```powershell
cd "C:\Users\zxc22\OneDrive\桌面\claude file\FactoryVision\work"; python stage3_sweep.py --list
```

```powershell
cd "C:\Users\zxc22\OneDrive\桌面\claude file\FactoryVision\work"; python -u stage3_sweep.py --only screen --epochs 100 > sweep.log 2>&1
```

```powershell
cd "C:\Users\zxc22\OneDrive\桌面\claude file\FactoryVision\work"; python stage3_sweep.py --table
```

本機預估：`screen`（五個 × 100 ep @512）約 **11 小時**，適合過夜。
`--only yolov8s_640_long`（250 ep @640）單獨跑約 **7 小時**。

跟 Colab 版一樣，每跑完一個就寫進 `work/outputs/stage3_sweep.json`，中斷不會白跑，
重跑會跳過已完成的。某個設定 OOM 也只會跳過它、繼續跑下一個。

---

## 兩邊都一樣的設計

- **同 seed（42）、同 augmentation、同 epoch 預算**，不然比較沒意義
- **`patience = epochs`**，sweep 期間不 early stop，每個設定都跑滿
- 評分用的是**我們自己實作的 VOC2010+ all-points mAP**（跟 TA 定義一致），
  不是 ultralytics 內建的那個。所以數字可以直接跟 0.7419 比
  （ultralytics 自己會算成 0.7588，比較寬鬆）
- conf 0.001 / NMS IoU 0.6 / max_det 100，跟繳交版一致

---

## 跑完回報給我

**最少給我這個**：`§6.7c`（或 `--table`）印出來的那段 JSON。

有了那個我會：

1. 確認哪個設定真的贏，特別看 crazing / rolled_in_scale 有沒有改善
2. 檢查是不是用更大的模型換來的——規格書 p.242 問「更大的 detector 是否真的值得」，
   延遲變太多反而是報告裡的扣分項
3. 要換 checkpoint 的話，我需要**那顆 `best.pt`**（Drive 上的 `<name>_best.pt`，
   放回 `work/` 底下就行），我會在本機重新量一次確認數字對得上
4. 然後更新：`Team13_Stage3/stage3_model.pt`、`config.json`、`stage3_report.pdf`、
   `Team13_Final_Report.pdf` 的 Stage 3 數字、重新打包 zip、跑兩支驗證器

**先不要自己改繳交包裡的東西**，等數字確認再一次改完，免得報告跟 checkpoint 不一致。
