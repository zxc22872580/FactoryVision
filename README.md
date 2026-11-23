# FactoryVision Challenge — 專案總覽

鋼材表面瑕疵辨識與定位（NEU-DET / NEU-CLS），三個 Stage + Final Report。

## 要繳交的東西

| 檔案 | 內容 | 大小 |
| --- | --- | --- |
| `Team13_Stage1.zip` | Baseline + Improved CNN | 3.1 MB |
| `Team13_Stage2.zip` | Transfer Learning | 40.3 MB |
| `Team13_Stage3.zip` | Object Detection | 24.0 MB |
| `Team13_Final_Report.pdf` | 期末總結報告 | 5.2 MB |

> **上傳前要做的事**：把 `Team13` 全部換成你的組別編號（兩位數，例如 `Team03`）。
> 需要改的地方：三個資料夾名稱、三個 zip 檔名、`Team13_Final_Report.pdf` 檔名，
> 以及三個 `config.json` 裡的 `"team_id"` 欄位。改完後重新壓縮，
> 並用 `work/check_submission.py` 再跑一次確認。

## 各 Stage 的最終模型與成績

| Stage | Final Model | Public Validation | 備註 |
| --- | --- | --- | --- |
| 1 | Baseline CNN + Data Augmentation + BatchNorm（from scratch，615,974 參數） | Macro F1 **1.0000**，劣化條件平均 0.7203 | 以 Train+Val 1,530 張重訓後提交 |
| 2 | ResNet18 + L2-SP（ImageNet pretrained） | Macro F1 **1.0000**，內部 Robust F1 0.9790 | 提交選模版（Train 1,260），未用重訓版 |
| 3 | YOLOv8s（COCO pretrained，640×640） | mAP@0.5 **0.7419**，35.3 ms/張 | conf 0.001 / NMS IoU 0.6 / max_det 100 |

依 TA 公告（三個 Stage 一併於期末繳交、Hidden Benchmark 由 TA 在收件後執行、
Final Report 僅需分析 training 結果），Final Report 的所有分析都建立在
Public Training / Validation 上，不含、也不需要 Hidden Benchmark 成績。

## 資料夾結構

```
FactoryVision/
├─ Team13_Stage1/          繳交內容（已通過 32 項格式檢查）
├─ Team13_Stage2/          繳交內容（33 項）
├─ Team13_Stage3/          繳交內容（35 項）
├─ Team13_Final_Report.pdf
├─ Team13_Stage{1,2,3}.zip
└─ work/                   不繳交，重現實驗用
   ├─ analyze_split.py        公開 split 的近重複分析
   ├─ validate_submission_official.py  課程官方提供的結構 validator
   ├─ check_submission.py     加強版檢查（會實際執行固定指令並驗證輸出 schema）
   ├─ package_stage.py        產生 config.json / README / 複製 checkpoint
   ├─ make_stage{1,2,3}_report.py, make_final_report.py
   ├─ plots.py, report_builder.py
   ├─ run_stage2.sh, run_stage3.sh, wait_and_eval.sh, run_stage3_eval.py
   ├─ stage1_runs/, stage2_runs/, stage3_runs/   訓練輸出與 summary JSON
   ├─ outputs/               所有指標的 JSON（報告的唯一資料來源）
   ├─ figures/               報告用圖
   └─ det_dataset/           Ultralytics 格式的偵測資料（由公開 split 複製）
```

## 繳交前自我檢查

**課程官方 validator**（TA 公告附的那支，三個 Stage 目前皆 `RESULT: PASS`，無任何 WARN）：

```bash
cd "C:/Users/zxc22/OneDrive/桌面/claude file/FactoryVision" && for s in 1 2 3; do python work/validate_submission_official.py --stage $s --submission "Team13_Stage$s"; done
```

**我們自己的加強版檢查**（官方只做結構與語法檢查，這支會真的從另一個工作目錄
執行固定指令、逐項驗證輸出 schema，三個 Stage 分別 32 / 33 / 35 項全過）：

```bash
cd work
python check_submission.py --stage 1 --submission ../Team13_Stage1 --sample_dir mock_hidden
python check_submission.py --stage 2 --submission ../Team13_Stage2 --sample_dir mock_hidden
python check_submission.py --stage 3 --submission ../Team13_Stage3 --sample_dir mock_det
```

## 環境

Python 3.11.9、PyTorch 2.6.0+cu124、torchvision 0.21.0、ultralytics 8.4.215、
scikit-learn 1.9.1。訓練在 GTX 1650 Ti（4 GB）上完成。

所有實驗固定 `random seed = 42`，使用課程提供的固定 split，未做任何重新切分。
