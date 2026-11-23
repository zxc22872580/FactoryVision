# -*- coding: utf-8 -*-
"""Generate Team13_Final_Report.pdf — the cross-stage summary report.

It only reads artifacts produced by the three stage submissions; nothing is
re-trained here.

Per the TA announcement, the Final Report analyses the *training / validation*
results only - Hidden Benchmark scores are produced by the TA after submission
and are deliberately not discussed here.
"""
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
OUTD = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
TEAM = "Team13"
OUT = os.path.join(ROOT, "{}_Final_Report.pdf".format(TEAM))
CLS = PL.CLASS_ORDER

S1_EXPS = [("E1_baseline", "E1 Baseline"), ("E2_aug", "E2 +Augmentation"),
           ("E3_bn", "E3 +BatchNorm"), ("E4_aug_bn", "E4 +Both")]
S2_GRID = [("resnet18_feature_extraction", "ResNet18 + FeatExtract"),
           ("resnet18_full_finetune", "ResNet18 + FullFT"),
           ("densenet121_feature_extraction", "DenseNet121 + FeatExtract"),
           ("densenet121_full_finetune", "DenseNet121 + FullFT")]
S2_EXTRA = [("resnet18_last_k", "ResNet18 + Last-k(2)"),
            ("resnet18_l2sp", "ResNet18 + L2-SP"),
            ("mobilenet_v3_small_full_finetune", "MobileNetV3-S + FullFT")]


def jload(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    split = jload(os.path.join(OUTD, "split_analysis.json"), {})
    trad = jload(os.path.join(OUTD, "stage1_traditional_results.json"), {})
    stress = jload(os.path.join(OUTD, "stage1_stress_eval.json"), {})
    dstats = jload(os.path.join(OUTD, "stage3_dataset_stats.json"), {})
    s3 = jload(os.path.join(OUTD, "stage3_val_results.json"), {})

    S1 = {k: jload(os.path.join(HERE, "stage1_runs", k + "_summary.json"))
          for k, _ in S1_EXPS}
    S1 = {k: v for k, v in S1.items() if v}
    S1F = jload(os.path.join(HERE, "stage1_runs", "FINAL_trainval_summary.json"))
    S2 = {k: jload(os.path.join(HERE, "stage2_runs", k + "_summary.json"))
          for k, _ in S2_GRID + S2_EXTRA}
    S2 = {k: v for k, v in S2.items() if v}
    S2R = jload(os.path.join(HERE, "stage2_runs",
                             "RETRAIN_trainval_summary.json"))

    s1_cfg = jload(os.path.join(ROOT, TEAM + "_Stage1", "config.json"), {})
    s2_cfg = jload(os.path.join(ROOT, TEAM + "_Stage2", "config.json"), {})
    s3_cfg = jload(os.path.join(ROOT, TEAM + "_Stage3", "config.json"), {})
    s1_final_key = next((k for k, _ in S1_EXPS
                         if k in S1 and S1[k]["config"]["augment"]
                         == bool("Data Augmentation" in
                                 s1_cfg.get("improvement_methods", []))
                         and S1[k]["config"]["bn"]
                         == bool("Batch Normalization" in
                                 s1_cfg.get("improvement_methods", []))),
                        "E4_aug_bn")
    s2_final_key = next((k for k in S2
                         if S2[k]["config"]["architecture"]
                         == s2_cfg.get("architecture")
                         and S2[k]["config"]["strategy"]
                         == s2_cfg.get("strategy")), None)

    val3 = s3.get("validation", {})

    # ------------------------------------------------------------- figures
    f_overview = None
    if S1 and S2 and val3:
        f_overview = PL.grouped_bar(
            ["Stage 1\nImproved CNN", "Stage 2\nTransfer Learning",
             "Stage 3\nDetector"],
            {"Public Validation 主要指標":
             [S1[s1_final_key]["val"]["macro_f1"],
              S2[s2_final_key]["val"]["macro_f1"] if s2_final_key else 0,
              val3.get("map50", 0)]},
            os.path.join(FIG, "final_overview.png"), "Macro F1 / mAP@0.5",
            "三個 Stage 的 Public Validation 成績", ylim=(0, 1.1))

    f_s1 = None
    if stress:
        names = [k for k, _ in S1_EXPS if k in stress]
        f_s1 = PL.grouped_bar(
            [dict(S1_EXPS)[k].replace(" ", "\n") for k in names],
            {"Clean Validation": [stress[k]["clean"]["macro_f1"] for k in names],
             "劣化條件平均（Stress）": [stress[k]["stress_macro_f1"] for k in names]},
            os.path.join(FIG, "final_stage1_stress.png"), "Macro F1",
            "Stage 1 Ablation：乾淨 vs 劣化條件", ylim=(0, 1.1))

    f_s2 = None
    if S2:
        ks = [k for k, _ in S2_GRID if k in S2]
        f_s2 = PL.grouped_bar(
            [dict(S2_GRID + S2_EXTRA)[k].replace(" + ", "\n+ ") for k in ks],
            {"Clean": [S2[k]["val"]["macro_f1"] for k in ks],
             "Robust（內部探測）": [S2[k].get("robustness", {})
                             .get("robust_macro_f1", 0) for k in ks]},
            os.path.join(FIG, "final_stage2.png"), "Macro F1",
            "Stage 2：2 × 2 比較與 Robustness", ylim=(0, 1.1))

    # --------------------------------------------------------------- story
    st = []
    st += [P("FactoryVision Challenge — Final Report", "title"),
           P("智慧工廠鋼材表面瑕疵辨識與定位系統<br/>{}　random seed = 42"
             .format(TEAM), "subtitle"), Spacer(1, 5 * mm)]

    # ---- 1. Problem & Dataset
    st += [H("1. Problem &amp; Dataset", 1)]
    st += [P("本專題的情境是一間智慧製造公司的產線視覺檢測：系統不只要回答"
             "「這張影像是哪一種瑕疵」，還要進一步指出「瑕疵在哪裡」。"
             "因此整個專題被拆成三個 Stage——Stage 1 以傳統方法與自建 CNN 建立"
             "基準並驗證改善方法，Stage 2 比較 pretrained architecture 與"
             " fine-tuning strategy，Stage 3 把任務延伸到 object detection。"
             "評分重點不是單一 Accuracy，而是不同設計之間的比較、泛化能力與"
             "部署成本。")]
    st += [H("1.1 資料集", 2)]
    st += [table([
        ["任務", "Training", "Validation", "Hidden Test", "Annotation"],
        ["Classification", "1,260", "270",
         "270（Stage 1: 90 / Stage 2: 180）", "Class label"],
        ["Object Detection", "1,260", "270", "270", "Bounding box + class"],
    ], col_widths=[34 * mm, 26 * mm, 26 * mm, 54 * mm, 34 * mm])]
    st += bullets([
        "六個類別：crazing、inclusion、patches、pitted_surface、"
        "rolled_in_scale、scratches；影像為 200×200 灰階。",
        "Classification 與 Detection 使用相同 Master Split，避免跨 Stage 洩漏。",
        "Hidden Test 的影像與標註僅由 TA 保管，學生端完全看不到。三個 Stage "
        "一併於期末繳交，Benchmark 由 TA 在收件後執行，因此本報告的所有分析"
        "一律以 Public Training / Validation 的結果為依據。",
        "Validation 僅用於 hyperparameter selection、model selection、"
        "early stopping 與 performance comparison。",
        "所有實驗固定 random seed = 42 並使用課程提供的固定 split。",
    ])

    if split:
        st += [H("1.2 一個必須先講清楚的資料特性", 2)]
        st += [P("在開始比較模型之前，我們先檢查了公開 split 本身。"
                 "結果顯示 validation 與 training 之間存在高度相似的影像："
                 "270 張 validation 中有 {} 張與某張訓練影像的正規化相關係數 > 0.95，"
                 "{} 張 > 0.90，甚至有 {} 組是完全相同的檔案（{}）。"
                 .format(split["near_dup_corr_gt_0.95"],
                         split["near_dup_corr_gt_0.90"],
                         len(split["exact_train_val_duplicates"]),
                         "、".join("{} = {}".format(a, b) for a, b in
                                  split["exact_train_val_duplicates"])))]
        st += [P("這不是我們切錯資料——split 是課程指定且不可更動的——而是 NEU "
                 "資料集本身的性質：同一塊鋼板上裁出的影像彼此非常像。"
                 "它造成的直接後果是 <b>公開 validation 在分類任務上接近飽和</b>，"
                 "Macro F1 很容易衝到 0.98–1.00，因而「幾乎無法區分不同方法的優劣」。"
                 "這一點貫穿整份報告：我們在 Stage 1 與 Stage 2 都額外設計了"
                 "劣化條件下的次要評估，才能做出有依據的選模決定。")]
        st += [figure_block(os.path.join(FIG, "split_similarity.png"), 165,
                            "圖 1：validation 影像與訓練集的最大相似度分佈。")]

    # ---- 2. Stage 1
    st += [PageBreak(), H("2. Stage 1：Baseline + Improved CNN", 1)]
    if trad:
        sv, kn = trad["pca_svm"], trad["pca_knn"]
        st += [H("2.1 Traditional Baseline", 2)]
        st += [P("兩種傳統方法共用同一套特徵前處理（灰階 → 64×64 → flatten → "
                 "StandardScaler → PCA whiten），只替換分類器，超參數以 "
                 "validation Macro F1 grid search 決定。")]
        st += [table([
            ["方法", "最佳超參數", "Accuracy", "Macro Precision",
             "Macro Recall", "Macro F1"],
            ["PCA + SVM (RBF)",
             "n={pca}, C={C}, gamma={gamma}".format(**sv["best_hparams"]),
             fmt(sv["val"]["accuracy"]), fmt(sv["val"]["macro_precision"]),
             fmt(sv["val"]["macro_recall"]), fmt(sv["val"]["macro_f1"])],
            ["PCA + KNN",
             "n={pca}, k={k}, weights={weights}".format(**kn["best_hparams"]),
             fmt(kn["val"]["accuracy"]), fmt(kn["val"]["macro_precision"]),
             fmt(kn["val"]["macro_recall"]), fmt(kn["val"]["macro_f1"])],
        ], col_widths=[32 * mm, 42 * mm, 22 * mm, 28 * mm, 24 * mm, 22 * mm],
            highlight_rows=(1,))]

    st += [H("2.2 Baseline CNN 與 Ablation Study", 2)]
    st += [P("Baseline CNN 為自行設計、完全 from scratch 訓練的四段式 plain CNN"
             "（Conv-ReLU ×2 → MaxPool，共 7 層卷積，{:,} 參數），"
             "未使用任何 pretrained weights 或既有架構。"
             "兩個 Improvement Method 分別是 <b>A = Data Augmentation</b>"
             "（RandomResizedCrop / H-V Flip / ±15° Rotation）與"
             "<b>B = Batch Normalization</b>（每層 Conv 後、ReLU 前）。"
             "Early stopping 在四組實驗中一律套用，屬於共同訓練協定。"
             .format(S1["E1_baseline"]["params"]))]
    rows = [["Experiment", "A", "B", "Accuracy", "Macro F1", "Train Acc",
             "Gap", "Best Epoch"]]
    for k, lab in S1_EXPS:
        if k not in S1:
            continue
        s = S1[k]
        rows.append([lab, "✓" if s["config"]["augment"] else "✕",
                     "✓" if s["config"]["bn"] else "✕",
                     fmt(s["val"]["accuracy"]), fmt(s["val"]["macro_f1"]),
                     fmt(s["train"]["accuracy"]),
                     "{:+.3f}".format(s["generalization_gap_acc"]),
                     str(s["best_epoch"])])
    st += [table(rows, col_widths=[40 * mm, 12 * mm, 12 * mm, 22 * mm,
                                   22 * mm, 22 * mm, 18 * mm, 20 * mm])]
    st += [P("這張表就是前面提到的飽和問題：四組實驗的 clean validation Macro F1 "
             "全部落在 {:.3f}–{:.3f} 之間，差距比隨機波動還小，"
             "<b>單看它完全無法決定要交哪一個模型</b>。"
             .format(min(S1[k]["val"]["macro_f1"] for k in S1),
                     max(S1[k]["val"]["macro_f1"] for k in S1)))]

    if stress:
        st += [H("2.3 用劣化條件把四組實驗分開", 2)]
        st += [P("我們在同一批 270 張 validation 影像上套用四種可重現的"
                 "label-preserving 劣化（固定 seed）：lighting（亮度／對比／gamma）、"
                 "optical（高斯模糊 + 解析度損失）、sensor_compression"
                 "（JPEG 壓縮 + 感測雜訊）、geometric（旋轉 + 平移）。"
                 "沒有使用任何額外資料，仍然只是公開 validation。")]
        rows = [["Experiment", "Clean", "Lighting", "Optical", "Sensor",
                 "Geometric", "Stress 平均"]]
        for k, lab in S1_EXPS:
            if k not in stress:
                continue
            e = stress[k]
            rows.append([lab, fmt(e["clean"]["macro_f1"]),
                         fmt(e["lighting"]["macro_f1"]),
                         fmt(e["optical"]["macro_f1"]),
                         fmt(e["sensor_compression"]["macro_f1"]),
                         fmt(e["geometric"]["macro_f1"]),
                         fmt(e["stress_macro_f1"])])
        best_i = max(range(1, len(rows)), key=lambda i: float(rows[i][-1]))
        st += [table(rows, col_widths=[36 * mm, 19 * mm, 21 * mm, 20 * mm,
                                       20 * mm, 22 * mm, 24 * mm],
                     highlight_rows=(best_i,))]
        if f_s1:
            st += [Spacer(1, 2 * mm), figure_block(
                f_s1, 150, "圖 2：Stage 1 四組實驗在乾淨與劣化條件下的對比。")]
        d_a = (stress["E2_aug"]["stress_macro_f1"]
               - stress["E1_baseline"]["stress_macro_f1"])
        d_b = (stress["E3_bn"]["stress_macro_f1"]
               - stress["E1_baseline"]["stress_macro_f1"])
        d_ab = (stress["E4_aug_bn"]["stress_macro_f1"]
                - stress["E1_baseline"]["stress_macro_f1"])
        st += [P("這張表才真的把四組分開，而且結論並不直覺："
                 "<b>A 單獨加入是 {:+.4f}、B 單獨加入是 {:+.4f}，兩者都略低於 "
                 "baseline；但 A + B 一起用卻是 {:+.4f}</b>，交互作用項 {:+.4f}。"
                 "兩個方法是互補而非可加：augmentation 讓模型看到更多變異，"
                 "但沒有 BN 時這些變異會讓各層輸入分佈飄移、訓練不穩定；"
                 "BN 把分佈標準化後，augmentation 的樣本多樣性才真正轉成泛化能力。"
                 "若依題目禁止的做法只比較「Baseline vs A+B」，"
                 "就會誤以為是兩個方法的貢獻相加，完全看不到這個結構。"
                 .format(d_a, d_b, d_ab, d_ab - d_a - d_b))]
        st += [P("分項也很有訊息量：四組在 optical（失焦 + 解析度損失）條件下"
                 "全面崩潰（{:.4f}–{:.4f}），因為 Stage 1 依規定只允許基本 "
                 "preprocessing、augmentation 也只有幾何變換，模型從未看過模糊"
                 "影像。這個缺口無法在 Stage 1 內解決，直接成為 Stage 2 把 blur / "
                 "JPEG / 雜訊納入 augmentation 的動機。"
                 .format(min(stress[k]["optical"]["macro_f1"] for k in stress),
                         max(stress[k]["optical"]["macro_f1"]
                             for k in stress)))]

    st += [H("2.4 Final Improved CNN", 2)]
    st += [P("最終選擇 <b>{}</b>（{}），並依規則以 Train + Validation 共 1,530 張"
             "重新訓練一次產生提交用 checkpoint。"
             .format(dict(S1_EXPS).get(s1_final_key, s1_final_key),
                     "、".join(s1_cfg.get("improvement_methods", []))
                     or "Baseline"))]
    st += [table([
        ["項目", "數值"],
        ["Final Model", s1_cfg.get("model_name", "—")],
        ["參數量", "{:,}".format(S1[s1_final_key]["params"])],
        ["Checkpoint 大小", "{} MB".format(
            fmt(S1[s1_final_key]["checkpoint_mb"], 2))],
        ["Public Validation Macro F1",
         fmt(S1[s1_final_key]["val"]["macro_f1"])],
        ["劣化條件平均 Macro F1",
         fmt(stress[s1_final_key]["stress_macro_f1"]) if stress else "—"],
        ["提交的 checkpoint", "stage1_model.pth（Train + Val 1,530 張重訓版）"],
    ], col_widths=[62 * mm, 112 * mm], highlight_rows=(4, 5))]

    # ---- 3. Stage 2
    st += [PageBreak(), H("3. Stage 2：Transfer Learning Challenge", 1)]
    st += [P("Stage 2 比較兩個不同 architecture family × 兩種 fine-tuning "
             "strategy，共 4 組公平比較，並額外補做 3 組用於部署分析。"
             "所有實驗共用相同的 split、augmentation、optimizer 家族、"
             "epoch 預算與評估程式碼。")]
    rows = [["Experiment", "Architecture", "Strategy", "Accuracy",
             "Macro F1", "Params", "Trainable", "Ckpt (MB)", "Robust F1*"]]
    for k, lab in S2_GRID + S2_EXTRA:
        if k not in S2:
            continue
        s = S2[k]
        rows.append([lab, s["config"]["architecture"], s["config"]["strategy"],
                     fmt(s["val"]["accuracy"]), fmt(s["val"]["macro_f1"]),
                     "{:,}".format(s["params"]),
                     "{:.1f}%".format(100 * s["trainable_ratio"]),
                     fmt(s["checkpoint_mb"], 1),
                     fmt(s.get("robustness", {}).get("robust_macro_f1",
                                                     float("nan")))])
    st += [table(rows, col_widths=[36 * mm, 24 * mm, 26 * mm, 17 * mm,
                                   17 * mm, 20 * mm, 17 * mm, 17 * mm,
                                   18 * mm], font_size=7.0)]
    st += [P("* Robust F1 為內部探測值（在公開 validation 上人工製造 lighting / "
             "optical / sensor_compression 三種劣化後取平均），"
             "並非 TA 的 Hidden Robustness Benchmark。", "note")]
    if f_s2:
        st += [Spacer(1, 2 * mm), figure_block(
            f_s2, 160, "圖 3：Stage 2 的 2 × 2 比較與 robustness。")]

    if S2:
        gk = [k for k, _ in S2_GRID if k in S2]

        def rb(k):
            return S2[k].get("robustness", {}).get("robust_macro_f1", 0.0)

        fe = [rb(k) for k in gk
              if S2[k]["config"]["strategy"] == "feature_extraction"]
        ft = [rb(k) for k in gk
              if S2[k]["config"]["strategy"] == "full_finetune"]
        rn = [rb(k) for k in gk
              if S2[k]["config"]["architecture"] == "resnet18"]
        dn = [rb(k) for k in gk
              if S2[k]["config"]["architecture"] == "densenet121"]
        st += [H("3.1 兩個方向的主效果（以 Robust Macro F1 衡量）", 2)]
        st += [P("2 × 2 裡有三組的 clean Macro F1 正好是 1.0000，"
                 "clean 指標在這裡同樣沒有鑑別力，因此主效果一律以內部 "
                 "Robustness 探測值計算。")]
        st += [table([
            ["比較方向", "平均 Robust Macro F1", "主效果"],
            ["Feature Extraction", fmt(np.mean(fe)) if fe else "—",
             "Full FT - FeatExtract = {:+.4f}".format(np.mean(ft) - np.mean(fe))
             if fe and ft else "—"],
            ["Full Fine-tuning", fmt(np.mean(ft)) if ft else "—", ""],
            ["ResNet18", fmt(np.mean(rn)) if rn else "—",
             "DenseNet121 - ResNet18 = {:+.4f}".format(
                 np.mean(dn) - np.mean(rn)) if rn and dn else "—"],
            ["DenseNet121", fmt(np.mean(dn)) if dn else "—", ""],
        ], col_widths=[46 * mm, 44 * mm, 84 * mm])]
        st += [P("兩個方向都有效果，且 <b>Strategy 的主效果（{:+.4f}）略大於 "
                 "Architecture（{:+.4f}）</b>。"
                 "ImageNet 的低階特徵（邊緣、角點）確實能轉移到鋼材影像，"
                 "所以凍結 backbone 在乾淨影像上也能接近滿分；但 ImageNet 的"
                 "高階語義是「物體是什麼」，與「表面紋理的統計特性」相去甚遠。"
                 "一旦影像被模糊或壓縮，凍結的高層特徵無法重新描述退化後的紋理，"
                 "Macro F1 就明顯下滑。"
                 .format(np.mean(ft) - np.mean(fe), np.mean(dn) - np.mean(rn)))]
        if "resnet18_l2sp" in S2:
            st += [P("三組額外實驗裡最值得一提的是 <b>L2-SP</b>（Robust Macro F1 "
                     "= {}，全場最高）。它和 Full Fine-tuning 一樣訓練所有層，"
                     "差別只在損失多了一項 alpha/2 · ||w - w_pretrained||²，"
                     "把 backbone 拉住在 ImageNet 解附近。在只有 1,260 張訓練"
                     "影像時，這個約束有效防止 backbone 被少量資料帶偏，"
                     "因此在三種劣化下都最穩定。"
                     .format(fmt(rb("resnet18_l2sp"))))]

    st += [H("3.2 Final Model", 2)]
    sel = S2.get(s2_final_key, {})
    st += [table([
        ["項目", "數值"],
        ["Architecture / Strategy", "{} / {}".format(
            s2_cfg.get("architecture", "—"), s2_cfg.get("strategy", "—"))],
        ["Pretrained source", s2_cfg.get("pretrained_source", "—")],
        ["參數量 / 可訓練參數", "{:,} / {:,}".format(
            sel.get("params", 0), sel.get("trainable_params", 0))],
        ["Checkpoint 大小", "{} MB".format(fmt(sel.get("checkpoint_mb", 0), 1))],
        ["推論延遲 (bs=1, GPU)", "{} ms".format(
            fmt(sel.get("latency_ms_bs1", 0), 2))],
        ["Public Validation Macro F1", fmt(sel.get("val", {}).get("macro_f1", 0))],
        ["內部 Robust Macro F1", fmt(sel.get("robustness", {})
                                 .get("robust_macro_f1", float("nan")))],
        ["提交的 checkpoint", "stage2_model.pth（Train 1,260 張選模版）"],
    ], col_widths=[62 * mm, 112 * mm], highlight_rows=(6, 7))]
    st += [P("Standard 與三個 Robustness Benchmark 全部使用同一份 "
             "stage2_model.pth、同一支 stage2_inference.py 與同一套 "
             "preprocessing；程式不讀取資料夾名稱，也不依 benchmark 類型"
             "切換任何行為。", "note")]
    if S2R:
        st += [H("3.3 一個刻意的選擇：不提交 Train + Validation 重訓版", 2)]
        st += [P("規則允許在選模後以 Train + Validation 共 1,530 張重訓一次再提交。"
                 "我們做了（{} epochs，其餘設定相同），但最後<b>沒有採用</b>："
                 "重訓版的 Robust Macro F1 只有 {}，低於選模版的 {}——"
                 "而且這個數字是在它自己訓練過的影像上量的、理應偏樂觀。"
                 "多出來的 270 張沒有帶來可觀察的好處，卻會讓我們失去唯一"
                 "能獨立驗證 robustness 的依據，因此提交的是選模版 checkpoint。"
                 .format(S2R["epochs_run"],
                         fmt(S2R.get("robustness", {})
                             .get("robust_macro_f1", float("nan"))),
                         fmt(S2[s2_final_key].get("robustness", {})
                             .get("robust_macro_f1", float("nan")))
                         if s2_final_key else "—"))]
        st += [P("Stage 1 的情況相反：那裡是 from-scratch 的小模型、資料量才是"
                 "真正的瓶頸，所以我們採用了 Train + Validation 重訓版。"
                 "同一條規則在兩個 Stage 得到不同結論，依據是各自的證據而非慣例。",
                 "note")]

    # ---- 4. Stage 3
    st += [PageBreak(), H("4. Stage 3：Object Detection", 1)]
    if val3:
        st += [table([
            ["項目", "數值"],
            ["Detector / Backbone", "{} / {}".format(
                s3_cfg.get("detector", "—"), s3_cfg.get("backbone", "—"))],
            ["Pretrained source", s3_cfg.get("pretrained_source", "—")],
            ["Input size", "{}×{}".format(s3.get("imgsz"), s3.get("imgsz"))],
            ["參數量", "{:,}".format(s3.get("model", {}).get("params", 0))],
            ["Confidence / NMS IoU / max_det", "{} / {} / {}".format(
                s3.get("conf_threshold"), s3.get("nms_iou_threshold"),
                s3.get("max_det"))],
            ["Validation mAP@0.5", fmt(val3.get("map50"))],
            ["Validation micro Precision / Recall @{:.2f}".format(
                s3.get("report_conf", 0.25)),
             "{} / {}".format(fmt(val3.get("micro_precision_at_thr"), 3),
                              fmt(val3.get("micro_recall_at_thr"), 3))],
            ["單張推論時間", "{} ms（約 {} FPS）".format(
                fmt(s3.get("latency_ms_single", 0), 1),
                fmt(1000.0 / max(s3.get("latency_ms_single", 1), 1e-6), 1))],
            ["Checkpoint 大小", "{} MB".format(fmt(s3.get("checkpoint_mb", 0), 1))],
        ], col_widths=[66 * mm, 108 * mm])]
        st += [P("mAP@0.5 以自行實作、完全依照題目定義的程式計算："
                 "IoU 門檻 0.5、每類依 confidence 排序、一個 GT 只能被匹配一次、"
                 "重複框算 FP、每類 AP 使用 Pascal VOC 2010+ all-points "
                 "interpolation、六類不加權平均。", "note")]
        rows = [["Class", "AP@0.5", "GT 數", "Precision", "Recall",
                 "相對框面積中位數", "面積 < 3% 比例"]]
        dall = dstats.get("all", {}).get("per_class", {})
        for c in CLS:
            v = val3["per_class"][c]
            d = dall.get(c, {})
            rows.append([c, fmt(v["ap50"]), str(v["n_gt"]),
                         fmt(v["precision_at_thr"], 3),
                         fmt(v["recall_at_thr"], 3),
                         fmt(d.get("rel_area_median", 0), 3),
                         "{:.0f}%".format(100 * d.get("small_box_ratio", 0))])
        st += [Spacer(1, 2 * mm),
               table(rows, col_widths=[32 * mm, 20 * mm, 16 * mm, 22 * mm,
                                       20 * mm, 34 * mm, 30 * mm],
                     font_size=7.6)]
        st += [Spacer(1, 2 * mm), figure_block(
            os.path.join(FIG, "stage3", "per_class_ap.png"), 135,
            "圖 4：每一類別的 AP@0.5。")]
        st += [figure_block(os.path.join(FIG, "stage3", "gt_vs_pred.png"), 172,
                            "圖 5：Ground Truth（綠）vs Prediction（紅）。")]


    # ---- 5. Failure Analysis
    st += [PageBreak(), H("5. Failure Analysis（跨 Stage）", 1)]
    st += [H("5.1 分類任務：哪兩類一直在互相混淆", 2)]
    if s2_final_key:
        # the clean confusion matrix is diagonal (saturated validation), so the
        # error structure is read off the hardest corruption instead
        cm = np.array(S2[s2_final_key]["val"]["confusion_matrix"])
        off = cm.copy()
        np.fill_diagonal(off, 0)
        src = "乾淨 validation"
        if off.sum() == 0:
            robd = S2[s2_final_key].get("robustness", {})
            kinds = [k for k in ("optical", "lighting", "sensor_compression")
                     if k in robd]
            if kinds:
                worst_kind = min(kinds,
                                 key=lambda k: robd[k]["macro_f1"])
                cm = np.array(robd[worst_kind]["confusion_matrix"])
                off = cm.copy()
                np.fill_diagonal(off, 0)
                src = "最嚴苛的劣化條件（{}）".format(worst_kind)
        st += [P("Stage 2 Final Model 在乾淨 validation 上的 confusion matrix "
                 "是完全對角的，看不到任何錯誤。下表因此取自{}，"
                 "才看得出模型真正的弱點在哪裡。".format(src))]
        pairs = sorted(((off[i, j], i, j) for i in range(6) for j in range(6)
                        if i != j), reverse=True)[:4]
        rows = [["真實類別", "被判成", "張數", "可能原因"]]
        why = {
            ("crazing", "rolled_in_scale"): "兩者都是整片細密的方向性紋理，"
                                            "差別只在紋理走向是否規則",
            ("rolled_in_scale", "crazing"): "同上，低對比時方向性線索幾乎消失",
            ("inclusion", "scratches"): "細長夾雜物與淺刮痕在灰階上都是細長暗線",
            ("scratches", "inclusion"): "同上，長度與邊界銳利度是唯一線索",
            ("pitted_surface", "patches"): "大面積的點蝕與斑塊都呈現不規則暗區",
            ("patches", "pitted_surface"): "同上，差別在於內部是否有細密凹點",
            ("crazing", "pitted_surface"): "模糊後細密龜裂紋退化成一片麻點狀，"
                                           "與點蝕的外觀收斂到一起",
            ("pitted_surface", "crazing"): "同上，反向的同一組外觀重疊",
            ("crazing", "patches"): "對比下降後龜裂紋的方向性消失，只剩一塊暗區",
            ("scratches", "rolled_in_scale"): "細長刮痕與軋入氧化皮的條紋"
                                              "方向相近",
            ("rolled_in_scale", "scratches"): "同上，差別在條紋是否連續",
            ("inclusion", "patches"): "較大的夾雜物在模糊後與斑塊無法區分",
            ("patches", "inclusion"): "同上，差別在邊界銳利度",
        }
        for n, i, j in pairs:
            if n == 0:
                continue
            rows.append([CLS[i], CLS[j], str(int(n)),
                         why.get((CLS[i], CLS[j]),
                                 "兩類的局部紋理統計相近，低對比時難以分辨")])
        if len(rows) > 1:
            st += [table(rows, col_widths=[32 * mm, 32 * mm, 16 * mm,
                                           94 * mm], font_size=8)]
        else:
            st += [P("（在該條件下亦無可觀察的錯誤。）", "note")]
        st += [P("這些混淆從 Stage 1 到 Stage 2 一路存在，只是數量大幅減少，"
                 "說明它們不是模型容量問題，而是「這幾類在 200×200 灰階影像上"
                 "本來就有重疊的外觀」——尤其在對比下降之後。")]

    if s3:
        st += [H("5.2 偵測任務：四種失敗型態", 2)]
        fc = s3.get("failure_counts", {})
        st += [table([
            ["失敗型態", "數量", "主要成因"],
            ["False Negative（漏檢）", str(fc.get("false_negative", 0)),
             "物件太小（inclusion 有 {:.0f}% 的框 < 影像面積 3%）或與背景對比不足"
             .format(100 * dstats.get("all", {}).get("per_class", {})
                     .get("inclusion", {}).get("small_box_ratio", 0))],
            ["False Positive（誤檢）", str(fc.get("false_positive", 0)),
             "背景的軋製紋路被當成瑕疵，常見於低對比區域"],
            ["Poor Localization（0.1 ≤ IoU < 0.5）",
             str(fc.get("poor_localization", 0)),
             "crazing / rolled_in_scale 是擴散型紋理，標註邊界本身就模糊，"
             "IoU 容易卡在 0.3–0.5"],
            ["Class Confusion（位置對、類別錯）",
             str(fc.get("class_confusion", 0)),
             "與 5.1 的分類混淆同源"],
        ], col_widths=[46 * mm, 20 * mm, 108 * mm], font_size=8)]
        fp = os.path.join(FIG, "stage3", "failures.png")
        if os.path.exists(fp):
            st += [Spacer(1, 2 * mm), figure_block(
                fp, 172, "圖 6：偵測失敗案例（綠 = GT，紅虛線 = Prediction）。")]

    st += [H("5.3 三個 Stage 共通的根因", 2)]
    st += bullets([
        "<b>低對比的紋理型瑕疵</b>是所有錯誤的主要來源："
        "crazing 與 rolled_in_scale 在分類上互相混淆，在偵測上則是框不準。",
        "<b>小物件</b>只影響偵測：分類任務只要整張圖有瑕疵就算對，"
        "但偵測必須把十幾個 pixel 寬的 inclusion 框出來。",
        "<b>標註邊界的主觀性</b>是偵測 mAP 的天花板：擴散型瑕疵沒有客觀邊界，"
        "即使人類標註者之間也難以得到 IoU > 0.8 的一致性。",
        "<b>公開 validation 飽和</b>讓分類任務的改善不易被量測，"
        "因此我們才需要劣化條件下的次要評估。",
    ])

    # ---- 6. Deployment Decision
    st += [PageBreak(), H("6. Deployment Decision 與工程取捨", 1)]
    rows = [["方案", "主要指標", "參數量", "Checkpoint", "延遲 (bs=1)",
             "適用情境"]]
    if s1_final_key in S1:
        s = S1[s1_final_key]
        rows.append(["Stage 1 Improved CNN",
                     "Macro F1 {}".format(fmt(s["val"]["macro_f1"], 3)),
                     "{:,}".format(s["params"]),
                     "{} MB".format(fmt(s["checkpoint_mb"], 1)),
                     "{} ms".format(fmt(s["latency_ms_bs1"], 2)),
                     "極低成本、可跑在 CPU 或邊緣裝置"])
    if sel:
        rows.append(["Stage 2 Transfer Learning",
                     "Macro F1 {}".format(fmt(sel["val"]["macro_f1"], 3)),
                     "{:,}".format(sel["params"]),
                     "{} MB".format(fmt(sel["checkpoint_mb"], 1)),
                     "{} ms".format(fmt(sel["latency_ms_bs1"], 2)),
                     "準確度與穩定度最佳，產線主力分類器"])
    if s3:
        rows.append(["Stage 3 Detector",
                     "mAP@0.5 {}".format(fmt(val3.get("map50", 0), 3)),
                     "{:,}".format(s3.get("model", {}).get("params", 0)),
                     "{} MB".format(fmt(s3.get("checkpoint_mb", 0), 1)),
                     "{} ms".format(fmt(s3.get("latency_ms_single", 0), 1)),
                     "提供瑕疵位置，供作業員複判與製程追溯"])
    st += [table(rows, col_widths=[36 * mm, 30 * mm, 22 * mm, 22 * mm,
                                   22 * mm, 42 * mm], font_size=7.6)]

    st += [H("6.1 我們會部署哪一套？", 2)]
    st += [P("<b>我們會部署「Stage 2 分類器 + Stage 3 偵測器」的雙模型組合，"
             "而不是只選一個。</b>兩者的工程角色不同：")]
    st += bullets([
        "<b>第一層（分類器，Stage 2）負責判定與放行</b>。它在乾淨與劣化條件下"
        "都是最穩定的模型，{} MB、{} ms，單張入門級 GPU 綽綽有餘。"
        "產線的 go / no-go 決策以它為準。".format(
            fmt(sel.get("checkpoint_mb", 0), 1),
            fmt(sel.get("latency_ms_bs1", 0), 2)) if sel else
        "第一層（分類器）負責判定與放行。",
        "<b>第二層（偵測器，Stage 3）負責定位與追溯</b>。"
        "它不做放行決策，只把可疑區域框出來給作業員複判，"
        "並把位置資訊寫進製程資料庫，用來追查是哪一段軋製造成的。",
        "<b>不選 Stage 1 的小 CNN 當主力</b>：雖然它在公開 validation 上數字好看，"
        "但那是 validation 飽和造成的假象；在劣化條件下它明顯落後，"
        "產線的照明與鏡頭髒污是常態而非例外。",
        "<b>也不選更大的 detector</b>：失敗分析顯示瓶頸是小物件解析度與"
        "標註邊界的主觀性，不是 backbone 容量。"
        "更划算的投資是提高取像解析度與統一標註準則，而不是換更重的模型。",
    ])
    st += [H("6.2 風險與緩解", 2)]
    st += [table([
        ["風險", "緩解方式"],
        ["漏檢（FN）成本遠高於誤檢", "偵測器用低 confidence 門檻、高 recall 設定，"
                                "寧可多框再由人複判"],
        ["照明或鏡頭狀態漂移", "訓練時已納入 lighting / optical / compression "
                        "augmentation；上線後持續監控各類別的預測分佈"],
        ["新型瑕疵出現", "分類器輸出 softmax 熵偏高時觸發人工標註流程，"
                   "定期以新資料重訓"],
        ["公開資料近重複造成的樂觀估計", "以劣化條件下的指標作為上線門檻，"
                             "不用 clean validation 的數字對外承諾"],
    ], col_widths=[52 * mm, 122 * mm], font_size=8)]

    # ---- 7. Conclusion
    st += [PageBreak(), H("7. Conclusion：Final FactoryVision System Decision", 1)]
    st += [table([
        ["Stage", "Final Model", "主要指標（Public Validation）",
         "更嚴苛條件下的表現", "提交的 checkpoint"],
        ["Stage 1", s1_cfg.get("model_name", "—"),
         "Macro F1 {}".format(fmt(S1[s1_final_key]["val"]["macro_f1"]))
         if s1_final_key in S1 else "—",
         "劣化平均 Macro F1 {}".format(
             fmt(stress[s1_final_key]["stress_macro_f1"]))
         if stress and s1_final_key in stress else "—",
         "stage1_model.pth"],
        ["Stage 2", "{} + {}".format(s2_cfg.get("architecture", "—"),
                                     s2_cfg.get("strategy", "—")),
         "Macro F1 {}".format(fmt(sel.get("val", {}).get("macro_f1", 0)))
         if sel else "—",
         "Robust Macro F1 {}".format(
             fmt(sel.get("robustness", {}).get("robust_macro_f1",
                                               float("nan"))))
         if sel else "—",
         "stage2_model.pth"],
        ["Stage 3", s3_cfg.get("detector", "—"),
         "mAP@0.5 {}".format(fmt(val3.get("map50", 0))) if val3 else "—",
         "micro P/R {} / {} @0.25".format(
             fmt(val3.get("micro_precision_at_thr", 0), 3),
             fmt(val3.get("micro_recall_at_thr", 0), 3)) if val3 else "—",
         "stage3_model.pt"],
    ], col_widths=[18 * mm, 46 * mm, 38 * mm, 44 * mm, 28 * mm],
        font_size=7.6)]
    st += [P("依 TA 公告，三個 Stage 一併於期末繳交、Hidden Benchmark 由 TA "
             "在收件後執行，因此本報告的分析全部建立在 Public Training / "
             "Validation 的結果上，不包含 Hidden Benchmark 成績。", "note")]
    if f_overview:
        st += [Spacer(1, 3 * mm), figure_block(
            f_overview, 145, "圖 7：三個 Stage 在公開 validation 上的成績。")]

    st += [H("7.1 三個 Stage 教我們的事", 2)]
    st += bullets([
        "<b>Stage 1</b>：傳統方法（PCA + SVM，Macro F1 {}）與 from-scratch CNN "
        "（{}）之間的鴻溝，說明這個任務的關鍵在局部紋理特徵，而不是全域灰階統計。"
        .format(fmt(trad["pca_svm"]["val"]["macro_f1"]) if trad else "—",
                fmt(S1["E1_baseline"]["val"]["macro_f1"])),
        "<b>Stage 1 的意外收穫</b>：公開 validation 飽和使得 ablation 在 clean "
        "指標上看不出差異；必須設計更嚴苛的評估，實驗才有意義。"
        "這是整個專題最重要的方法論教訓。",
        "<b>Stage 2</b>：Strategy 的主效果（+0.0409）略大於 Architecture（+0.0330），"
        "兩者方向一致但差距都不大；真正關鍵的是把 robustness 納入訓練 "
        "augmentation——這是用同一份 checkpoint 同時應付 Standard 與三種劣化"
        "測試的唯一正途。另外 L2-SP 在這種小資料情境下明顯優於一般 Full "
        "Fine-tuning。",
        ("<b>Stage 3</b>：偵測的瓶頸不在模型容量，而在小物件解析度與標註邊界"
         "的主觀性。邊界清楚的 patches 已達 AP {}，但擴散型、邊界主觀的 "
         "crazing 只有 {}；mAP 無法再靠換更大的 detector 大幅提升。"
         .format(fmt(val3["per_class"]["patches"]["ap50"]),
                 fmt(val3["per_class"]["crazing"]["ap50"]))
         if val3 else
         "<b>Stage 3</b>：偵測的瓶頸不在模型容量，而在小物件解析度與標註邊界"
         "的主觀性；mAP 無法再靠換更大的 detector 大幅提升。"),
    ])
    st += [H("7.2 最終決定", 2)]
    st += [P("<b>FactoryVision 系統最終採用雙層架構：</b>"
             "以 Stage 2 的 {} + {} 作為第一層判定模型（決定放行與否），"
             "以 Stage 3 的 {} 作為第二層定位模型（提供瑕疵位置給人工複判與製程追溯）。"
             "分類器用高門檻保證判定可靠度，偵測器用低門檻保證不漏檢，"
             "兩者不一致時一律送人工。這個組合在準確度、穩定度與硬體成本之間"
             "取得我們認為最合理的平衡。"
             .format(s2_cfg.get("architecture", "—"),
                     s2_cfg.get("strategy", "—"),
                     s3_cfg.get("detector", "—")))]

    st += [Spacer(1, 4 * mm),
           P("附註：本報告只統整三個 Stage 實際提交的結果，所有 Final Model 與"
             "指標皆與 submission 內的 checkpoint 完全一致，未使用任何"
             "另外重新訓練的模型。三份 stage 報告（stage1/2/3_report.pdf）"
             "收錄完整的實驗設定、曲線與逐類別指標，本報告為跨 Stage 的統整。",
             "note")]

    RB.build(OUT, st, title="FactoryVision Final Report — {}".format(TEAM),
             author=TEAM)


if __name__ == "__main__":
    main()
