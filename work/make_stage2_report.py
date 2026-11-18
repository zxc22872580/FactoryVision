# -*- coding: utf-8 -*-
"""Generate TeamXX_Stage2/stage2_report.pdf from the experiment artifacts."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plots as PL
import report_builder as RB
from report_builder import (H, P, PageBreak, Spacer, bullets, figure_block,
                            fmt, mm, table)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "stage2_runs")
FIG = os.path.join(HERE, "figures", "stage2")
OUT = os.path.join(ROOT, "TeamXX_Stage2", "stage2_report.pdf")
TEAM = "TeamXX"

GRID = [("resnet18_feature_extraction", "E1 ResNet18 + Feature Extraction",
         "ResNet18", "Feature Extraction"),
        ("resnet18_full_finetune", "E2 ResNet18 + Full Fine-tuning",
         "ResNet18", "Full Fine-tuning"),
        ("densenet121_feature_extraction",
         "E3 DenseNet121 + Feature Extraction", "DenseNet121",
         "Feature Extraction"),
        ("densenet121_full_finetune", "E4 DenseNet121 + Full Fine-tuning",
         "DenseNet121", "Full Fine-tuning")]
EXTRA = [("resnet18_last_k", "E5 ResNet18 + Last-k (k=2)", "ResNet18",
          "Last-k Fine-tuning (k=2)"),
         ("resnet18_l2sp", "E6 ResNet18 + L2-SP", "ResNet18",
          "L2-SP Regularized Fine-tuning"),
         ("mobilenet_v3_small_full_finetune",
          "E7 MobileNetV3-S + Full Fine-tuning", "MobileNetV3-Small",
          "Full Fine-tuning")]
CLS = PL.CLASS_ORDER
S2_LABEL = [(k, lab) for k, lab, _, _ in GRID + EXTRA]


def load(name):
    p = os.path.join(RUNS, name + "_summary.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    os.makedirs(FIG, exist_ok=True)
    S = {}
    for k, *_ in GRID + EXTRA:
        s = load(k)
        if s:
            S[k] = s
    grid = [(k, lab, a, st) for k, lab, a, st in GRID if k in S]
    extra = [(k, lab, a, st) for k, lab, a, st in EXTRA if k in S]
    allruns = grid + extra

    # The train+val retrain exists but is NOT the submitted checkpoint;
    # section 4.2 explains why.
    rpath = os.path.join(RUNS, "RETRAIN_trainval_summary.json")
    retrain = json.load(open(rpath, encoding="utf-8")) \
        if os.path.exists(rpath) else None
    sel_key = max(S, key=lambda k: (
        S[k].get("robustness", {}).get("robust_macro_f1", 0)
        + S[k]["val"]["macro_f1"]))
    sel_label = dict((k, lab) for k, lab, _, _ in allruns)[sel_key]

    def rob(k, kind=None):
        r = S[k].get("robustness")
        if not r:
            return float("nan")
        return r["robust_macro_f1"] if kind is None else r[kind]["macro_f1"]

    # ------------------------------------------------------------- figures
    f_curv_r = PL.training_curves(
        [S[k] for k, _, a, _ in grid if a == "ResNet18"],
        [lab for _, lab, a, _ in grid if a == "ResNet18"],
        os.path.join(FIG, "curves_resnet18.png"), "ResNet18 訓練曲線")
    f_curv_d = PL.training_curves(
        [S[k] for k, _, a, _ in grid if a == "DenseNet121"],
        [lab for _, lab, a, _ in grid if a == "DenseNet121"],
        os.path.join(FIG, "curves_densenet121.png"), "DenseNet121 訓練曲線")
    f_grid = PL.grouped_bar(
        ["ResNet18", "DenseNet121"],
        {"Feature Extraction": [S[k]["val"]["macro_f1"] for k, _, _, st in grid
                                if st == "Feature Extraction"],
         "Full Fine-tuning": [S[k]["val"]["macro_f1"] for k, _, _, st in grid
                              if st == "Full Fine-tuning"]},
        os.path.join(FIG, "grid_f1.png"), "Validation Macro F1",
        "2 × 2 公平比較：Architecture × Strategy", ylim=(0, 1.08))
    f_rob = PL.grouped_bar(
        [lab.split(" ", 1)[0] for _, lab, _, _ in allruns],
        {"Clean": [S[k]["val"]["macro_f1"] for k, _, _, _ in allruns],
         "Lighting": [rob(k, "lighting") for k, _, _, _ in allruns],
         "Optical": [rob(k, "optical") for k, _, _, _ in allruns],
         "Sensor/Compression": [rob(k, "sensor_compression")
                                for k, _, _, _ in allruns]},
        os.path.join(FIG, "robustness.png"), "Macro F1",
        "內部 Robustness 探測（公開 validation 加人工劣化）",
        annotate=False, ylim=(0, 1.08))
    f_cm = PL.confusion_matrix_fig(
        S[sel_key]["val"]["confusion_matrix"],
        os.path.join(FIG, "cm_final.png"),
        "Final Classification Model Confusion Matrix")
    f_trade = PL.scatter_tradeoff(
        [(S[k]["checkpoint_mb"], S[k]["val"]["macro_f1"],
          lab.split(" ", 1)[0]) for k, lab, _, _ in allruns],
        os.path.join(FIG, "tradeoff.png"), "Checkpoint 大小 (MB)",
        "Validation Macro F1", "部署成本 vs 準確度")

    # --------------------------------------------------------------- story
    st = []
    st += [P("FactoryVision Challenge — Stage 2 報告", "title"),
           P("Transfer Learning Challenge｜Architecture × Fine-tuning Strategy"
             "<br/>{}　random seed = 42".format(TEAM), "subtitle"),
           Spacer(1, 6 * mm)]

    st += [H("1. 實驗設計", 1)]
    st += [P("Stage 2 的目標是在同一份 Classification Public Dataset 上，比較不同"
             " pretrained architecture family 與不同 fine-tuning strategy 的"
             "組合，並選出最適合部署的分類模型。除了 Standard Benchmark，"
             "TA 還會以 lighting / optical / sensor_compression 三種劣化版本測試"
             " Robustness，而且四次測試必須使用同一個 checkpoint 與同一支"
             " inference script，因此「對劣化影像的穩定度」從一開始就被納入設計。")]
    st += bullets([
        "Architecture Family A：ResNet18（residual family）。",
        "Architecture Family B：DenseNet121（dense-connectivity family）。",
        "Strategy 1：Feature Extraction — 凍結整個 backbone，只訓練新的 classifier head。",
        "Strategy 2：Full Fine-tuning — 所有參數皆可訓練。",
        "以上 2 × 2 = 4 組為必要的公平比較；另外補做 Last-k、L2-SP 與 MobileNetV3-Small 共 3 組，用於部署分析。",
        "所有實驗共用完全相同的 split、augmentation pipeline、optimizer 家族、epoch 預算、early-stopping 規則與評估程式碼，唯一變動的只有 architecture 與 strategy。",
    ])
    cfg0 = S[grid[0][0]]["config"]
    st += [table([
        ["共同設定", "數值"],
        ["Pretrained source", cfg0["pretrained_source"]],
        ["Input size", "{}×{}（灰階複製成 3 通道）".format(cfg0["input_size"],
                                                 cfg0["input_size"])],
        ["Normalization", "ImageNet mean {} / std {}".format(
            cfg0["norm_mean"], cfg0["norm_std"])],
        ["Optimizer", "AdamW，weight_decay = {}".format(cfg0["weight_decay"])],
        ["Learning rate", "Feature Extraction 1e-3／其餘 1e-4，CosineAnnealingLR"],
        ["Batch size / Epochs", "{} / 最多 {}（early stopping patience = 8）"
            .format(cfg0["batch_size"], cfg0["epochs"])],
        ["Loss", "CrossEntropyLoss，label_smoothing = {}".format(
            cfg0["label_smoothing"])],
        ["Model selection", "validation Macro F1 最高的 epoch"],
        ["Seed", str(cfg0["seed"])],
    ], col_widths=[42 * mm, 132 * mm])]

    st += [H("1.1 共用的 Augmentation Pipeline", 2)]
    aug = cfg0.get("robust_augmentation") or {}
    st += [P("為了同時服務 Standard 與三種 Robustness Benchmark，訓練時套用的"
             "augmentation 直接對應三種劣化類型。這個 pipeline 在四組實驗中完全相同，"
             "因此不會影響公平比較；inference 端則一律只做 Resize / ToTensor / "
             "Normalize，對所有輸入資料夾一視同仁。")]
    st += [table([["對應的 Benchmark", "Augmentation", "參數"]] + [
        ["lighting", "ColorJitter",
         "brightness={b}, contrast={c}, p={p}".format(
             b=aug.get("color_jitter", {}).get("brightness"),
             c=aug.get("color_jitter", {}).get("contrast"),
             p=aug.get("color_jitter", {}).get("p"))],
        ["lighting", "RandomGamma", "range={r}, p={p}".format(
            r=tuple(aug.get("random_gamma", {}).get("range", ())),
            p=aug.get("random_gamma", {}).get("p"))],
        ["optical", "GaussianBlur", "kernel={k}, sigma={s}, p={p}".format(
            k=aug.get("gaussian_blur", {}).get("kernel"),
            s=tuple(aug.get("gaussian_blur", {}).get("sigma", ())),
            p=aug.get("gaussian_blur", {}).get("p"))],
        ["optical", "Downscale→Upscale", "factor={f}, p={p}".format(
            f=tuple(aug.get("downscale_upscale", {}).get("factor", ())),
            p=aug.get("downscale_upscale", {}).get("p"))],
        ["sensor_compression", "JPEG re-compression",
         "quality={q}, p={p}".format(
             q=tuple(aug.get("jpeg_compression", {}).get("quality", ())),
             p=aug.get("jpeg_compression", {}).get("p"))],
        ["sensor_compression", "Gaussian noise", "std={s}, p={p}".format(
            s=tuple(aug.get("gaussian_noise", {}).get("std", ())),
            p=aug.get("gaussian_noise", {}).get("p"))],
        ["一般泛化", "RandomResizedCrop / Flip / Rotation / RandomErasing",
         "scale={s}, rotation=±{r}°, erasing p={e}".format(
             s=tuple(aug.get("random_resized_crop_scale", ())),
             r=aug.get("random_rotation_degrees"),
             e=aug.get("random_erasing", {}).get("p"))],
    ], col_widths=[34 * mm, 58 * mm, 82 * mm])]

    # ------------------------------------------------------------ 2x2 table
    st += [PageBreak(), H("2. 2 × 2 公平比較結果", 1)]
    rows = [["Experiment", "Architecture", "Strategy", "Accuracy",
             "Macro Precision", "Macro Recall", "Macro F1", "Best Epoch",
             "訓練秒數"]]
    for k, lab, a, stg in grid:
        s = S[k]
        rows.append([lab, a, stg, fmt(s["val"]["accuracy"]),
                     fmt(s["val"]["macro_precision"]),
                     fmt(s["val"]["macro_recall"]), fmt(s["val"]["macro_f1"]),
                     str(s["best_epoch"]), "{:.0f}".format(s["train_seconds"])])
    st += [table(rows, col_widths=[44 * mm, 22 * mm, 26 * mm, 17 * mm,
                                   20 * mm, 17 * mm, 16 * mm, 13 * mm,
                                   14 * mm], font_size=7.4)]
    st += [Spacer(1, 3 * mm), figure_block(
        f_grid, 140, "圖 1：2 × 2 公平比較的 validation Macro F1。")]

    st += [H("2.1 Parameter / Trainable Parameter / Model Size", 2)]
    rows = [["Experiment", "Total Params", "Trainable Params", "Trainable 比例",
             "Checkpoint (MB)", "推論延遲 bs=1 (ms)"]]
    for k, lab, _, _ in allruns:
        s = S[k]
        rows.append([lab, "{:,}".format(s["params"]),
                     "{:,}".format(s["trainable_params"]),
                     "{:.2f}%".format(100 * s["trainable_ratio"]),
                     fmt(s["checkpoint_mb"], 1),
                     fmt(s["latency_ms_bs1"], 2)])
    st += [table(rows, col_widths=[48 * mm, 26 * mm, 28 * mm, 22 * mm,
                                   25 * mm, 28 * mm], font_size=7.6)]
    st += [Spacer(1, 2 * mm), figure_block(
        f_trade, 135, "圖 2：checkpoint 大小與 Macro F1 的取捨。")]

    st += [PageBreak(), H("2.2 Training / Validation Curves", 2)]
    st += [figure_block(f_curv_r, 175,
                        "圖 3：ResNet18 在兩種 strategy 下的訓練曲線。")]
    st += [figure_block(f_curv_d, 175,
                        "圖 4：DenseNet121 在兩種 strategy 下的訓練曲線。")]

    st += [H("2.3 Frozen / Trainable Layers 明細", 2)]
    rows = [["Experiment", "Frozen Layers", "Trainable Layers"]]
    for k, lab, _, _ in allruns:
        c = S[k]["config"]
        rows.append([lab,
                     "、".join(c["frozen_layers"]) or "（無）",
                     "、".join(c["trainable_layers"])])
    st += [table(rows, col_widths=[44 * mm, 62 * mm, 68 * mm], font_size=7.2)]
    if "resnet18_l2sp" in S:
        c = S["resnet18_l2sp"]["config"]
        st += [P("L2-SP 設定：alpha（backbone 對 pretrained 權重的 L2-SP 係數）= {}，"
                 "beta（新 head 的一般 L2 係數）= {}；此時 optimizer 的 weight_decay "
                 "設為 0，避免重複正則化。".format(c["l2sp_alpha"], c["l2sp_beta"]),
                 "note")]
    if "resnet18_last_k" in S:
        st += [P("Last-k 設定：k = {}，亦即只解凍 ResNet18 最後 {} 個 top-level stage"
                 "（layer3、layer4）加上新的 classifier head，其餘全部凍結。"
                 .format(S["resnet18_last_k"]["config"]["last_k"],
                         S["resnet18_last_k"]["config"]["last_k"]), "note")]

    # ------------------------------------------------------------ analysis
    st += [PageBreak(), H("3. 結果分析", 1)]
    st += [H("3.1 Clean Macro F1 在這個資料集上已經沒有鑑別力", 2)]
    st += [P("七組實驗裡有五組的 clean validation Macro F1 正好是 1.0000，"
             "其餘兩組也有 {:.4f}。公開 validation 與訓練集存在大量近重複影像"
             "（Final Report 第 1.2 節有完整分析），pretrained backbone 只要稍微"
             "fine-tune 就能全部答對。<b>因此本節的所有比較都以內部 Robustness "
             "探測值（Robust Macro F1）為主要依據</b>，clean 值只用來確認沒有"
             "模型明顯失敗。"
             .format(min(S[k]["val"]["macro_f1"] for k in S)))]
    fe_r = [rob(k) for k, _, _, stg in grid if stg == "Feature Extraction"]
    ft_r = [rob(k) for k, _, _, stg in grid if stg == "Full Fine-tuning"]
    rn_r = [rob(k) for k, _, a, _ in grid if a == "ResNet18"]
    dn_r = [rob(k) for k, _, a, _ in grid if a == "DenseNet121"]
    st += [table([
        ["比較方向", "平均 Clean Macro F1", "平均 Robust Macro F1", "主效果"],
        ["Feature Extraction",
         fmt(np.mean([S[k]["val"]["macro_f1"] for k, _, _, stg in grid
                      if stg == "Feature Extraction"])), fmt(np.mean(fe_r)),
         "Full FT - FeatExtract = {:+.4f}".format(np.mean(ft_r) - np.mean(fe_r))],
        ["Full Fine-tuning",
         fmt(np.mean([S[k]["val"]["macro_f1"] for k, _, _, stg in grid
                      if stg == "Full Fine-tuning"])), fmt(np.mean(ft_r)), ""],
        ["ResNet18",
         fmt(np.mean([S[k]["val"]["macro_f1"] for k, _, a, _ in grid
                      if a == "ResNet18"])), fmt(np.mean(rn_r)),
         "DenseNet121 - ResNet18 = {:+.4f}".format(np.mean(dn_r) - np.mean(rn_r))],
        ["DenseNet121",
         fmt(np.mean([S[k]["val"]["macro_f1"] for k, _, a, _ in grid
                      if a == "DenseNet121"])), fmt(np.mean(dn_r)), ""],
    ], col_widths=[42 * mm, 40 * mm, 42 * mm, 50 * mm], font_size=8)]
    st += [P("兩個方向都有效果，<b>Strategy 的主效果（{:+.4f}）略大於 Architecture "
             "（{:+.4f}）</b>，而且兩者方向一致：解凍 backbone 比凍結好，"
             "DenseNet121 比 ResNet18 稍好。"
             .format(np.mean(ft_r) - np.mean(fe_r), np.mean(dn_r) - np.mean(rn_r)))]
    st += [P("Strategy 的影響可以這樣理解：ImageNet 的低階特徵（邊緣、角點）"
             "確實能轉移到鋼材影像，所以 Feature Extraction 在乾淨影像上也能做到"
             "接近滿分；但 ImageNet 的高階語義是「物體是什麼」，與「表面紋理的"
             "統計特性」相去甚遠。一旦影像被模糊或壓縮，凍結的高層特徵無法重新"
             "描述退化後的紋理，Macro F1 就大幅下滑——"
             "{} 在 optical 條件下只剩 {:.4f} 就是最極端的例子。"
             .format(dict(S2_LABEL)["resnet18_feature_extraction"],
                     rob("resnet18_feature_extraction", "optical"))
             if "resnet18_feature_extraction" in S else "")]
    st += [P("Architecture 的差距則小得多。DenseNet121 的 dense connectivity 讓"
             "淺層的高解析度特徵能直接傳到分類層，對「細緻紋理」這種需要低階"
             "資訊的任務稍有優勢；而且它的參數量（{:,}）其實比 ResNet18（{:,}）"
             "還少，checkpoint 小了約 {:.0f}%。"
             .format(S["densenet121_full_finetune"]["params"],
                     S["resnet18_full_finetune"]["params"],
                     100 * (1 - S["densenet121_full_finetune"]["checkpoint_mb"]
                            / S["resnet18_full_finetune"]["checkpoint_mb"]))
           )] if "densenet121_full_finetune" in S else []

    st += [H("3.1.1 額外三組策略帶來的資訊", 2)]
    extra_rows = [["Experiment", "Clean", "Robust", "觀察"]]
    notes = {
        "resnet18_last_k": "只解凍 layer3 / layer4 + head，但那已佔 ResNet18 "
                           "{:.0f}% 的參數，效果因此接近 Full Fine-tuning".format(
                               100 * S["resnet18_last_k"]["trainable_ratio"])
        if "resnet18_last_k" in S else "",
        "resnet18_l2sp": "全層訓練但以 L2-SP 把權重拉住在 pretrained 附近，"
                         "robustness 最佳",
        "mobilenet_v3_small_full_finetune":
            "參數量只有 ResNet18 的 {:.0f}%、checkpoint {:.1f} MB，"
            "robustness 仍達 {:.4f}".format(
                100 * S["mobilenet_v3_small_full_finetune"]["params"]
                / S["resnet18_full_finetune"]["params"],
                S["mobilenet_v3_small_full_finetune"]["checkpoint_mb"],
                rob("mobilenet_v3_small_full_finetune"))
            if "mobilenet_v3_small_full_finetune" in S else "",
    }
    for k, lab, _, _ in extra:
        extra_rows.append([lab, fmt(S[k]["val"]["macro_f1"]), fmt(rob(k)),
                           notes.get(k, "")])
    st += [table(extra_rows, col_widths=[46 * mm, 18 * mm, 18 * mm, 92 * mm],
                 font_size=7.8)]
    st += [P("L2-SP 的結果值得特別說明：它和 Full Fine-tuning 一樣訓練所有層，"
             "差別只在損失函數多了一項 alpha/2 · ||w - w_pretrained||²，"
             "把 backbone 的權重拉住在 ImageNet 解附近。"
             "在只有 1,260 張訓練影像的情況下，這個約束有效防止 backbone 被"
             "少量資料帶偏，因此在三種劣化下都最穩定——"
             "這正是「資料量小 + 需要泛化」情境下 L2-SP 被提出的理由。")]

    st += [H("3.2 Robustness 探測", 2)]
    st += [P("TA 的 Robustness Benchmark 不公開，因此我們在公開 validation 上"
             "以可重現的方式人工製造三種劣化（固定 seed），作為選模的輔助依據。"
             "這只是內部估計，絕對數值不等於 Hidden Benchmark，但足以比較不同模型"
             "的相對穩定度。")]
    rows = [["Experiment", "Clean", "Lighting", "Optical",
             "Sensor/Comp.", "Robust Macro F1*", "相對 Clean 衰退"]]
    for k, lab, _, _ in allruns:
        c = S[k]["val"]["macro_f1"]
        r = rob(k)
        rows.append([lab, fmt(c), fmt(rob(k, "lighting")),
                     fmt(rob(k, "optical")), fmt(rob(k, "sensor_compression")),
                     fmt(r), "{:+.4f}".format(r - c)])
    st += [table(rows, col_widths=[44 * mm, 17 * mm, 19 * mm, 18 * mm,
                                   28 * mm, 26 * mm, 22 * mm], font_size=7.4)]
    st += [P("* Robust Macro F1 =（Lighting + Optical + Sensor/Compression）/ 3，"
             "與 Stage 2 計分公式相同。", "note")]
    st += [Spacer(1, 2 * mm), figure_block(
        f_rob, 160, "圖 5：各模型在乾淨與三種劣化條件下的 Macro F1。")]

    # ------------------------------------------------------- final decision
    st += [PageBreak(), H("4. Final Model Decision 與 Deployment Analysis", 1)]
    sel = S[sel_key]
    st += [P("綜合 Standard 與 Robustness 兩項指標，Stage 2 的 Final "
             "Classification Model 選擇 <b>{}</b>。".format(sel_label))]
    st += bullets([
        "Validation Macro F1 = {}，在所有實驗中屬於最高一群。".format(
            fmt(sel["val"]["macro_f1"])),
        "內部 Robust Macro F1 = {}，相對 clean 的衰退僅 {:+.4f}，是三種劣化下最穩定的模型。"
        .format(fmt(rob(sel_key)), rob(sel_key) - sel["val"]["macro_f1"]),
        "Checkpoint {} MB、推論延遲 {} ms（bs = 1, GPU），"
        "單張產線影像的視覺檢測預算綽綽有餘。".format(
            fmt(sel["checkpoint_mb"], 1), fmt(sel["latency_ms_bs1"], 2)),
        "參數量 {:,}，可訓練參數 {:,}。".format(sel["params"],
                                      sel["trainable_params"]),
    ])
    st += [Spacer(1, 2 * mm), figure_block(
        f_cm, 105, "圖 6：Final Classification Model 的 confusion matrix。")]

    st += [H("4.1 Deployment Analysis", 2)]
    rows = [["評估面向", "ResNet18 (Full FT)", "DenseNet121 (Full FT)",
             "MobileNetV3-S (Full FT)"]]
    keys = ["resnet18_full_finetune", "densenet121_full_finetune",
            "mobilenet_v3_small_full_finetune"]
    present = [k for k in keys if k in S]
    for field, getter in [
            ("Macro F1 (clean)", lambda s: fmt(s["val"]["macro_f1"])),
            ("Robust Macro F1", lambda s: fmt(
                s.get("robustness", {}).get("robust_macro_f1", float("nan")))),
            ("參數量", lambda s: "{:,}".format(s["params"])),
            ("Checkpoint (MB)", lambda s: fmt(s["checkpoint_mb"], 1)),
            ("延遲 bs=1 (ms)", lambda s: fmt(s["latency_ms_bs1"], 2)),
            ("訓練時間 (s)", lambda s: "{:.0f}".format(s["train_seconds"]))]:
        rows.append([field] + [getter(S[k]) for k in present])
    st += [table(rows, col_widths=[36 * mm, 44 * mm, 46 * mm, 48 * mm])]
    st += [P("工程取捨：若產線端是嵌入式裝置（Jetson / 工業 PC 無獨立顯卡），"
             "MobileNetV3-Small 的模型大小與延遲優勢非常明顯；但在本任務上，"
             "鋼材瑕疵的判斷高度依賴細緻紋理，輕量模型的 Macro F1 與 Robustness "
             "都明顯落後。由於 FactoryVision 的檢測站通常是固定機台、可以配置一張"
             "入門級 GPU，我們選擇以準確度與穩定度為主的模型；"
             "若日後要移到邊緣裝置，建議以此模型做 knowledge distillation，"
             "而不是直接換成輕量 backbone。")]

    st += [H("4.2 提交用 checkpoint：為什麼不交 Train + Validation 重訓版", 2)]
    if retrain:
        st += [P("規則允許在 model selection 完成後，以 Train + Validation 共 "
                 "1,530 張重新訓練一次再提交。我們實際做了這件事"
                 "（{} epochs，其餘超參數完全相同），但<b>最後決定不採用它</b>。"
                 .format(retrain["epochs_run"]))]
        st += [table([
            ["Checkpoint", "訓練資料", "Clean Macro F1", "Robust Macro F1",
             "這個數字可信嗎？"],
            ["resnet18_l2sp（提交用）", "Train 1,260",
             fmt(sel["val"]["macro_f1"]), fmt(rob(sel_key)),
             "可信：validation 未參與訓練"],
            ["RETRAIN（train+val）", "Train + Val 1,530",
             fmt(retrain["val"]["macro_f1"]),
             fmt(retrain.get("robustness", {}).get("robust_macro_f1",
                                                   float("nan"))),
             "不可信：評估影像已在訓練集內，理應偏樂觀"],
        ], col_widths=[42 * mm, 30 * mm, 28 * mm, 30 * mm, 44 * mm],
            highlight_rows=(1,), font_size=7.8)]
        st += [P("理由很直接：重訓版的 Robust Macro F1 只有 {}，"
                 "<b>比選模版的 {} 還低，而且這個數字是在它自己看過的影像上量的、"
                 "本來就應該偏高</b>。換句話說多出來的 270 張（+21%）沒有帶來"
                 "可觀察的好處，卻讓我們失去唯一能獨立驗證 robustness 的依據。"
                 "Stage 2 的 Benchmark 有 12 分、其中 6 分來自 Robustness，"
                 "在沒有任何證據顯示重訓更好的情況下，交出一個無法驗證的"
                 " checkpoint 是不划算的風險。"
                 .format(fmt(retrain.get("robustness", {})
                             .get("robust_macro_f1", float("nan"))),
                         fmt(rob(sel_key))))]
        st += [P("補充：Stage 1 的情況相反，我們在那裡採用了 Train + Validation "
                 "重訓版。差別在於 Stage 1 是 from-scratch 的小模型、資料量是"
                 "真正的瓶頸，多 21% 資料有實質幫助；Stage 2 的模型已經"
                 " ImageNet pretrained 且在 clean validation 上飽和，"
                 "額外資料的邊際效益接近零。", "note")]
    else:
        st += [P("提交的 stage2_model.pth 即為 model selection 階段選出的 "
                 "checkpoint，訓練資料為 Train 1,260 張，validation 未參與訓練，"
                 "因此報告中的 robustness 數字是獨立估計。")]
    st += [P("Standard 與三個 Robustness Benchmark 都使用這同一份 checkpoint。",
             "note")]

    st += [PageBreak(), H("5. 繳交內容與執行方式", 1)]
    st += [P("python stage2_inference.py --input_dir &lt;DIR&gt; "
             "--output_csv predictions.csv", "code")]
    st += bullets([
        "同一支 stage2_inference.py 與同一份 stage2_model.pth 連續跑 4 次"
        "（Standard / lighting / optical / sensor_compression），"
        "輸出格式皆為 image_id,prediction。",
        "程式不讀取 input_dir 名稱、不依 benchmark 類型切換 checkpoint、"
        "threshold、normalization 或 prediction rule。",
        "preprocessing 固定為：灰階 → 複製 3 通道 → Resize(224,224) → ToTensor "
        "→ ImageNet Normalize，對所有輸入一致套用。",
        "inference 為 eval mode + torch.no_grad()，不使用 TTA 或 Ensemble；"
        "pretrained weights 由 checkpoint 載入，執行期不連網。",
    ])

    RB.build(OUT, st, title="FactoryVision Stage 2 — {}".format(TEAM),
             author=TEAM)


if __name__ == "__main__":
    main()
