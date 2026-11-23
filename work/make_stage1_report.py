# -*- coding: utf-8 -*-
"""Generate Team13_Stage1/stage1_report.pdf from the experiment artifacts."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plots as PL
import report_builder as RB
from report_builder import H, P, PageBreak, Spacer, bullets, figure_block, fmt, mm, table

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "stage1_runs")
FIG = os.path.join(HERE, "figures", "stage1")
OUT = os.path.join(ROOT, "Team13_Stage1", "stage1_report.pdf")
TEAM = "Team13"

EXPS = [("E1_baseline", "E1 Baseline CNN", "○", "○"),
        ("E2_aug", "E2 Baseline + A", "●", "○"),
        ("E3_bn", "E3 Baseline + B", "○", "●"),
        ("E4_aug_bn", "E4 Baseline + A + B", "●", "●")]
CLS = PL.CLASS_ORDER


def load(name):
    with open(os.path.join(RUNS, name + "_summary.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(FIG, exist_ok=True)
    trad = json.load(open(os.path.join(HERE, "outputs",
                                       "stage1_traditional_results.json"),
                          encoding="utf-8"))
    S = {k: load(k) for k, _, _, _ in EXPS}
    stress = json.load(open(os.path.join(HERE, "outputs",
                                         "stage1_stress_eval.json"),
                            encoding="utf-8"))
    split = json.load(open(os.path.join(HERE, "outputs",
                                        "split_analysis.json"),
                           encoding="utf-8"))
    # clean validation is saturated, so ties are broken by the stress score
    final_key = max(S, key=lambda k: (round(S[k]["val"]["macro_f1"], 4),
                                      stress[k]["stress_macro_f1"]))
    final = S[final_key]
    final_label = dict((k, lab) for k, lab, _, _ in EXPS)[final_key]
    retrain_path = os.path.join(RUNS, "FINAL_trainval_summary.json")
    retrain = (json.load(open(retrain_path, encoding="utf-8"))
               if os.path.exists(retrain_path) else None)

    # ---------------------------------------------------------------- figures
    f_curves = PL.training_curves([S[k] for k, _, _, _ in EXPS],
                                  [lab for _, lab, _, _ in EXPS],
                                  os.path.join(FIG, "curves.png"),
                                  "Stage 1 Ablation：四組實驗的訓練曲線")
    f_abl = PL.ablation_bar([lab.split(" ", 1)[0] + "\n" + lab.split(" ", 1)[1]
                             for _, lab, _, _ in EXPS],
                            [S[k]["val"]["macro_f1"] for k, _, _, _ in EXPS],
                            os.path.join(FIG, "ablation.png"),
                            title="Validation Macro F1（Ablation）")
    f_cm = PL.confusion_matrix_fig(
        final["val"]["confusion_matrix"], os.path.join(FIG, "cm_final.png"),
        "Final Improved CNN（{}）Confusion Matrix".format(final_label))
    f_cm_base = PL.confusion_matrix_fig(
        S["E1_baseline"]["val"]["confusion_matrix"],
        os.path.join(FIG, "cm_baseline.png"), "E1 Baseline CNN Confusion Matrix")
    f_cm_svm = PL.confusion_matrix_fig(
        trad["pca_svm"]["val"]["confusion_matrix"],
        os.path.join(FIG, "cm_svm.png"), "PCA + SVM Confusion Matrix")
    f_pc = PL.per_class_f1_bar(
        [trad["pca_svm"]["val"]["per_class_f1"]] +
        [S[k]["val"]["per_class_f1"] for k, _, _, _ in EXPS],
        ["PCA+SVM"] + [lab.split(" ")[0] for _, lab, _, _ in EXPS],
        os.path.join(FIG, "per_class_f1.png"), "每一類別的 Validation F1")
    f_stress = PL.grouped_bar(
        [lab.split(" ", 1)[0] for _, lab, _, _ in EXPS],
        {"Clean": [stress[k]["clean"]["macro_f1"] for k, _, _, _ in EXPS],
         "Lighting": [stress[k]["lighting"]["macro_f1"] for k, _, _, _ in EXPS],
         "Optical": [stress[k]["optical"]["macro_f1"] for k, _, _, _ in EXPS],
         "Sensor": [stress[k]["sensor_compression"]["macro_f1"]
                    for k, _, _, _ in EXPS],
         "Geometric": [stress[k]["geometric"]["macro_f1"]
                       for k, _, _, _ in EXPS]},
        os.path.join(FIG, "stress.png"), "Macro F1",
        "四組實驗在四種劣化條件下的表現", annotate=False, ylim=(0, 1.08))
    f_stress_sum = PL.ablation_bar(
        [lab.replace(" ", chr(10), 1)
         for _, lab, _, _ in EXPS],
        [stress[k]["stress_macro_f1"] for k, _, _, _ in EXPS],
        os.path.join(FIG, "stress_summary.png"),
        ylabel="劣化條件平均 Macro F1",
        title="Stress Macro F1（四種劣化的平均）")
    f_gap = PL.grouped_bar(
        [lab.split(" ")[0] for _, lab, _, _ in EXPS],
        {"Train Accuracy": [S[k]["train"]["accuracy"] for k, _, _, _ in EXPS],
         "Val Accuracy": [S[k]["val"]["accuracy"] for k, _, _, _ in EXPS]},
        os.path.join(FIG, "gap.png"), "Accuracy",
        "Overfitting 檢查：Train vs Validation Accuracy", ylim=(0, 1.08))

    # ------------------------------------------------------------------ story
    st = []
    st += [P("FactoryVision Challenge — Stage 1 報告", "title"),
           P("Baseline + Improved CNN｜鋼材表面瑕疵分類<br/>"
             "{}　random seed = 42".format(TEAM), "subtitle"),
           Spacer(1, 6 * mm)]

    st += [H("1. 任務設定與資料", 1)]
    st += [P("本階段的任務是在 FactoryVision Public Classification Dataset 上，"
             "把 200×200 的灰階鋼材表面影像分到六個瑕疵類別："
             "crazing、inclusion、patches、pitted_surface、rolled_in_scale、scratches。"
             "資料採用課程提供的固定 split，訓練 1,260 張（每類 210 張）、"
             "validation 270 張（每類 45 張），類別完全平衡；"
             "Hidden Test 由 TA 保管，學生端不可見。")]
    st += bullets([
        "Validation 僅用於 hyperparameter selection、model selection 與 early stopping，不參與梯度更新。",
        "所有實驗固定 random seed = 42，並開啟 cudnn.deterministic，確保 ablation 之間可公平比較。",
        "Baseline 階段只允許 Resize / ToTensor / Normalize；Normalization 參數由訓練集自行統計得到 mean = 0.5070、std = 0.2132（單通道）。",
    ])

    st += [H("2. Part A-1：Traditional Baseline（PCA + SVM / PCA + KNN）", 1)]
    st += [P("傳統方法採用完全相同的特徵前處理，只替換最後的分類器，"
             "因此兩者差異純粹來自分類器本身：灰階影像 → resize 64×64 → "
             "flatten 成 4,096 維 → StandardScaler → PCA（whiten）→ 分類器。"
             "PCA 維度、SVM 的 C/gamma、KNN 的 k 與 weights 皆以 validation Macro F1 做 grid search。")]
    sv, kn = trad["pca_svm"], trad["pca_knn"]
    st += [table([
        ["方法", "最佳超參數", "PCA 解釋變異", "Accuracy", "Macro Precision",
         "Macro Recall", "Macro F1"],
        ["PCA + SVM (RBF)",
         "n={pca}, C={C}, gamma={gamma}".format(**sv["best_hparams"]),
         fmt(sv["pca_explained_variance_ratio"], 3),
         fmt(sv["val"]["accuracy"]), fmt(sv["val"]["macro_precision"]),
         fmt(sv["val"]["macro_recall"]), fmt(sv["val"]["macro_f1"])],
        ["PCA + KNN",
         "n={pca}, k={k}, weights={weights}".format(**kn["best_hparams"]),
         fmt(kn["pca_explained_variance_ratio"], 3),
         fmt(kn["val"]["accuracy"]), fmt(kn["val"]["macro_precision"]),
         fmt(kn["val"]["macro_recall"]), fmt(kn["val"]["macro_f1"])],
    ], col_widths=[32 * mm, 38 * mm, 20 * mm, 19 * mm, 24 * mm, 20 * mm, 19 * mm],
        highlight_rows=(1,))]
    st += [Spacer(1, 3 * mm), figure_block(
        f_cm_svm, 105,
        "圖 1：PCA + SVM 的 validation confusion matrix。")]
    st += [P("兩者差距非常大（Macro F1 {:.4f} vs {:.4f}）。原因在於 PCA 得到的是全域"
             "灰階強度的線性主成分，KNN 直接在這個歐式空間上比距離，對鋼材影像的"
             "整體亮度與對比變化極度敏感；SVM 的 RBF kernel 則能在同一個特徵空間上"
             "學出非線性邊界，對亮度位移有較好的容忍度。但兩者都沒有任何局部紋理的"
             "概念——而 crazing / rolled_in_scale 的差別正是局部紋理方向，"
             "這是後面 CNN 能大幅勝出的根本原因。"
             .format(sv["val"]["macro_f1"], kn["val"]["macro_f1"]))]

    st += [PageBreak(), H("3. Part A-2：Self-designed Baseline CNN", 1)]
    base = S["E1_baseline"]
    st += [P("Baseline CNN 完全自行設計並 from scratch 訓練，沒有載入任何 pretrained "
             "weights，也沒有使用 ResNet / VGG / MobileNet 的既有架構或 residual / "
             "inverted-residual 等結構。架構為四個 stage 的 plain CNN：")]
    st += [P("輸入 1×128×128<br/>"
             "stage1: Conv3×3(1→32) + ReLU + Conv3×3(32→32) + ReLU + MaxPool2 → 32×64×64<br/>"
             "stage2: Conv3×3(32→64) + ReLU + Conv3×3(64→64) + ReLU + MaxPool2 → 64×32×32<br/>"
             "stage3: Conv3×3(64→128) + ReLU + Conv3×3(128→128) + ReLU + MaxPool2 → 128×16×16<br/>"
             "stage4: Conv3×3(128→256) + ReLU + MaxPool2 → 256×8×8<br/>"
             "head  : AdaptiveAvgPool(1×1) → Flatten → Linear(256→128) + ReLU → Linear(128→6)",
             "code")]
    st += [table([
        ["項目", "設定"],
        ["Convolution / Activation / Pooling / Classifier", "皆具備（見上方架構）"],
        ["參數量", "{:,}（baseline）／{:,}（加入 BN 後）".format(
            base["params"], S["E3_bn"]["params"])],
        ["Optimizer", "Adam，lr = 1e-3，CosineAnnealingLR（T_max = 80）"],
        ["Loss", "CrossEntropyLoss"],
        ["Batch size / Epochs", "64 / 最多 80（early stopping patience = 20）"],
        ["Model selection", "以 validation Macro F1 最高的 epoch 為準"],
        ["推論延遲（batch=1, GPU）", "{} ms".format(fmt(base["latency_ms_bs1"], 2))],
        ["Checkpoint 大小", "{} MB".format(fmt(base["checkpoint_mb"], 2))],
    ], col_widths=[62 * mm, 112 * mm])]
    st += [Spacer(1, 3 * mm), figure_block(
        f_cm_base, 105,
        "圖 2：E1 Baseline CNN 的 validation confusion matrix。")]

    st += [PageBreak(), H("4. Part B：兩種 Improvement Method", 1)]
    st += [H("4.1 Method A — Data Augmentation", 2)]
    aug = S["E2_aug"]["config"]["aug_config"]
    st += [P("依題目 FAQ，整組 basic data augmentation pipeline 視為「一種」"
             "improvement method。完整設定如下（只作用於訓練集，validation 與 "
             "inference 一律使用基本 preprocessing）：")]
    st += [table([["Augmentation", "參數"]] + [
        ["RandomResizedCrop", "size=128, scale={}, ratio=(0.9, 1.111)".format(
            tuple(aug["random_resized_crop_scale"]))],
        ["RandomHorizontalFlip", "p = {}".format(aug["random_horizontal_flip_p"])],
        ["RandomVerticalFlip", "p = {}".format(aug["random_vertical_flip_p"])],
        ["RandomRotation", "degrees = ±{}".format(aug["random_rotation_degrees"])],
    ], col_widths=[52 * mm, 122 * mm])]
    st += [P("選擇理由：鋼材瑕疵沒有固定方向性，水平／垂直翻轉與小角度旋轉都是"
             "label-preserving 的；RandomResizedCrop 則模擬相機取景與瑕疵尺度的變化。"
             "刻意不加入顏色擾動，因為 Stage 1 的影像是單通道灰階且取像條件固定。", "note")]

    st += [H("4.2 Method B — Batch Normalization", 2)]
    st += [P("在每一層 Convolution 之後、ReLU 之前插入 BatchNorm2d（共 7 層），"
             "並把該層 Conv 的 bias 關閉（bias 由 BN 的 shift 取代）。"
             "momentum 與 eps 使用 PyTorch 預設值 0.1 / 1e-5，weight 初始化為 1、bias 為 0。"
             "額外參數僅 {:,} 個（{} 與 {} 之差），幾乎不影響模型大小與推論成本。"
             .format(S["E3_bn"]["params"] - base["params"],
                     "{:,}".format(S["E3_bn"]["params"]),
                     "{:,}".format(base["params"])))]
    st += [P("選擇理由：baseline 是沒有 residual connection 的 plain CNN，深層時梯度"
             "尺度不穩；BN 讓每層輸入分佈標準化，可以使用較大的有效學習率、"
             "同時提供輕微的正則化效果。", "note")]
    st += [P("說明：early stopping（patience = 20，監控 validation Macro F1）在 "
             "E1–E4 四組實驗中「全部」套用，屬於共同的訓練協定而非其中一個 "
             "improvement method，如此 ablation 的差異才完全來自 A 與 B。", "note")]

    # ------------------------------------------------------------- ablation
    st += [PageBreak(), H("5. Ablation Study（4 組實驗）", 1)]
    rows = [["Experiment", "A", "B", "Accuracy", "Macro<br/>Precision",
             "Macro<br/>Recall", "Macro F1", "Best<br/>Epoch",
             "Train<br/>Acc", "Gap"]]
    for k, lab, a, b in EXPS:
        s = S[k]
        rows.append([lab, a, b, fmt(s["val"]["accuracy"]),
                     fmt(s["val"]["macro_precision"]),
                     fmt(s["val"]["macro_recall"]),
                     fmt(s["val"]["macro_f1"]), str(s["best_epoch"]),
                     fmt(s["train"]["accuracy"]),
                     fmt(s["generalization_gap_acc"], 3)])
    hl = 1 + [k for k, _, _, _ in EXPS].index(final_key)
    st += [table(rows, col_widths=[36 * mm, 10 * mm, 10 * mm, 18 * mm, 19 * mm,
                                   17 * mm, 17 * mm, 14 * mm, 16 * mm, 17 * mm],
                 highlight_rows=(hl,), font_size=7.6)]
    st += [Spacer(1, 2 * mm),
           P("A = Data Augmentation，B = Batch Normalization；"
             "● 表示啟用、○ 表示未啟用。"
             "Gap = Train Accuracy - Validation Accuracy（訓練集一律以 evaluation "
             "transform 重新評估，不含 augmentation，才能公平看 overfitting）。", "note")]
    st += [Spacer(1, 2 * mm), figure_block(
        f_abl, 125, "圖 3：四組 ablation 的 validation Macro F1。")]
    st += [figure_block(f_curves, 175,
                        "圖 4：四組實驗的 loss / accuracy / Macro F1 曲線。"
                        "虛線為 training、實線為 validation，★ 為被選用的 epoch。")]
    st += [figure_block(f_gap, 135,
                        "圖 5：Train / Validation Accuracy 對照，用於觀察 overfitting。")]

    st += [H("5.1 一個必須先面對的問題：clean validation 已經飽和", 2)]
    st += [P("四組實驗的 validation Macro F1 全部落在 {:.4f}–{:.4f} 之間，"
             "連「完全沒加任何改善方法」的 E1 都達到 {:.4f}。"
             "也就是說，<b>這張 ablation 表單獨看並不能告訴我們哪個方法比較好</b>。"
             .format(min(S[k]["val"]["macro_f1"] for k in S),
                     max(S[k]["val"]["macro_f1"] for k in S),
                     base["val"]["macro_f1"]))]
    st += [P("我們回頭檢查了公開 split 本身，發現原因不在模型而在資料："
             "270 張 validation 中有 {} 張與某張訓練影像的正規化相關係數 > 0.95，"
             "{} 張 > 0.90，其中 {} 組甚至是位元組完全相同的檔案（{}）。"
             "這是 NEU 資料集本身的性質（同一塊鋼板裁切出的影像彼此極像），"
             "而 split 是課程指定、不可更動的，我們也沒有去更動它。"
             .format(split["near_dup_corr_gt_0.95"],
                     split["near_dup_corr_gt_0.90"],
                     len(split["exact_train_val_duplicates"]),
                     "、".join("{} = {}".format(a, b) for a, b in
                              split["exact_train_val_duplicates"])))]
    st += [Spacer(1, 2 * mm), figure_block(
        os.path.join(HERE, "figures", "split_similarity.png"), 160,
        "圖 6：validation 影像對訓練集的最大相似度分佈，"
        "右側長尾即為近重複影像。")]

    st += [PageBreak(), H("5.2 次要評估：在劣化條件下重新比較", 2)]
    st += [P("為了讓 ablation 真的有鑑別力，我們在<b>同一批 270 張 validation 影像</b>"
             "上套用四種可重現（固定 seed）、label-preserving 的劣化，再重新評估"
             "四個 checkpoint。沒有使用任何額外資料，也沒有重新切分；"
             "這仍然只是公開 validation，用途仍是規則允許的 model selection。")]
    st += [table([
        ["劣化類型", "內容", "模擬的產線狀況"],
        ["lighting", "亮度 ×0.45–1.6、對比 ×0.5–1.6、gamma 0.7–1.5",
         "照明老化、角度改變、反光"],
        ["optical", "高斯模糊 σ 0.8–2.2 + 降取樣至 35–65% 再放大",
         "鏡頭失焦、鏡頭髒污"],
        ["sensor_compression", "JPEG quality 12–35 + 高斯雜訊 σ 4–14",
         "影像傳輸壓縮、感測器雜訊"],
        ["geometric", "旋轉 ±25° + 平移 ±18 px",
         "鋼板擺放角度、相機未校正"],
    ], col_widths=[34 * mm, 78 * mm, 62 * mm])]
    rows = [["Experiment", "Clean", "Lighting", "Optical", "Sensor",
             "Geometric", "Stress 平均"]]
    for k, lab, _, _ in EXPS:
        e = stress[k]
        rows.append([lab, fmt(e["clean"]["macro_f1"]),
                     fmt(e["lighting"]["macro_f1"]),
                     fmt(e["optical"]["macro_f1"]),
                     fmt(e["sensor_compression"]["macro_f1"]),
                     fmt(e["geometric"]["macro_f1"]),
                     fmt(e["stress_macro_f1"])])
    hl2 = 1 + [k for k, _, _, _ in EXPS].index(
        max(stress, key=lambda k: stress[k]["stress_macro_f1"]))
    st += [Spacer(1, 2 * mm),
           table(rows, col_widths=[36 * mm, 19 * mm, 21 * mm, 20 * mm,
                                   20 * mm, 22 * mm, 24 * mm],
                 highlight_rows=(hl2,))]
    st += [Spacer(1, 2 * mm), figure_block(
        f_stress_sum, 125, "圖 7：四組實驗的 Stress Macro F1。")]
    st += [figure_block(f_stress, 165,
                        "圖 8：拆解到四種劣化條件的表現。")]

    d_a = stress["E2_aug"]["stress_macro_f1"] - stress["E1_baseline"]["stress_macro_f1"]
    d_b = stress["E3_bn"]["stress_macro_f1"] - stress["E1_baseline"]["stress_macro_f1"]
    d_ab = stress["E4_aug_bn"]["stress_macro_f1"] - stress["E1_baseline"]["stress_macro_f1"]
    c_a = S["E2_aug"]["val"]["macro_f1"] - base["val"]["macro_f1"]
    c_b = S["E3_bn"]["val"]["macro_f1"] - base["val"]["macro_f1"]
    c_ab = S["E4_aug_bn"]["val"]["macro_f1"] - base["val"]["macro_f1"]
    st += [H("5.3 個別效果與交互作用", 2)]
    st += [table([
        ["比較", "Clean Macro F1 變化", "Stress Macro F1 變化", "說明"],
        ["A 的單獨效果（E2 − E1）", "{:+.4f}".format(c_a),
         "{:+.4f}".format(d_a), "只加 Data Augmentation"],
        ["B 的單獨效果（E3 − E1）", "{:+.4f}".format(c_b),
         "{:+.4f}".format(d_b), "只加 Batch Normalization"],
        ["A + B（E4 − E1）", "{:+.4f}".format(c_ab),
         "{:+.4f}".format(d_ab), "兩者同時使用"],
        ["交互作用 = (E4−E1) − [(E2−E1)+(E3−E1)]",
         "{:+.4f}".format(c_ab - c_a - c_b), "{:+.4f}".format(d_ab - d_a - d_b),
         "> 0 表示互補"],
    ], col_widths=[58 * mm, 30 * mm, 30 * mm, 56 * mm], font_size=7.8)]
    st += [P("這張表是整個 Stage 1 最值得注意的結果：<b>A 與 B 各自單獨加入時，"
             "在劣化條件下反而都比 baseline 略差（{:+.4f} 與 {:+.4f}），"
             "但兩者併用卻帶來 {:+.4f} 的提升</b>，交互作用項高達 {:+.4f}。"
             "若只做「Baseline vs A+B」的兩組比較，會誤以為是兩個方法各自的"
             "貢獻相加——實際上它們是互補的：augmentation 讓模型看到更多變異，"
             "但沒有 BN 時這些變異會讓各層輸入分佈飄移、訓練不穩；"
             "BN 把分佈標準化之後，augmentation 的樣本多樣性才真正轉化成泛化能力。"
             "這正是題目要求四組 ablation、而非只比較頭尾兩組的理由。"
             .format(d_a, d_b, d_ab, d_ab - d_a - d_b))]
    st += [P("分項來看也符合預期：E4 在 geometric 上最強（{:.4f}），"
             "因為 Method A 的旋轉與翻轉正是針對幾何變異；"
             "四組在 optical 上都明顯偏弱（{:.4f}–{:.4f}），"
             "因為 Stage 1 的 augmentation 依規定只做幾何變換、沒有模糊，"
             "模型從未看過失焦影像。這個缺口正是 Stage 2 把 blur 與 JPEG "
             "納入 augmentation 的直接動機。"
             .format(stress["E4_aug_bn"]["geometric"]["macro_f1"],
                     min(stress[k]["optical"]["macro_f1"] for k in stress),
                     max(stress[k]["optical"]["macro_f1"] for k in stress)))]

    st += [PageBreak(), H("6. Final Improved CNN 的選擇", 1)]
    st += [P("選模準則分兩層：<b>先看 clean validation Macro F1，並列第一時再以"
             "劣化條件下的 Stress Macro F1 決勝</b>。"
             "E1 與 E4 的 clean Macro F1 同為 {}，但 E4 的 Stress Macro F1 為 {}、"
             "E1 只有 {}，差距 {:+.4f}，因此選出 <b>{}</b> 作為 Stage 1 的 "
             "Final Improved CNN，也是 Hidden Benchmark 使用的 checkpoint。"
             .format(fmt(final["val"]["macro_f1"]),
                     fmt(stress[final_key]["stress_macro_f1"]),
                     fmt(stress["E1_baseline"]["stress_macro_f1"]),
                     stress[final_key]["stress_macro_f1"]
                     - stress["E1_baseline"]["stress_macro_f1"],
                     final_label))]
    st += [P("補充一點誠實的說明：因為 clean validation 已經飽和，"
             "這個選擇的直接依據是劣化條件下的次要指標，而不是 clean Macro F1 的"
             "高低；若只看 clean 指標，E1 與 E4 是無法區分的。", "note")]
    st += [table([
        ["指標", "PCA + SVM", "E1 Baseline CNN", final_label + "（Final）"],
        ["Accuracy", fmt(sv["val"]["accuracy"]), fmt(base["val"]["accuracy"]),
         fmt(final["val"]["accuracy"])],
        ["Macro Precision", fmt(sv["val"]["macro_precision"]),
         fmt(base["val"]["macro_precision"]), fmt(final["val"]["macro_precision"])],
        ["Macro Recall", fmt(sv["val"]["macro_recall"]),
         fmt(base["val"]["macro_recall"]), fmt(final["val"]["macro_recall"])],
        ["Macro F1", fmt(sv["val"]["macro_f1"]), fmt(base["val"]["macro_f1"]),
         fmt(final["val"]["macro_f1"])],
        ["參數量", "—", "{:,}".format(base["params"]),
         "{:,}".format(final["params"])],
        ["Checkpoint 大小", "—", "{} MB".format(fmt(base["checkpoint_mb"], 2)),
         "{} MB".format(fmt(final["checkpoint_mb"], 2))],
        ["推論延遲 (bs=1, GPU)", "{} ms".format(fmt(
            sv["val_inference_ms_per_image"], 2)),
         "{} ms".format(fmt(base["latency_ms_bs1"], 2)),
         "{} ms".format(fmt(final["latency_ms_bs1"], 2))],
    ], col_widths=[40 * mm, 38 * mm, 44 * mm, 50 * mm], highlight_rows=(4,))]
    st += [Spacer(1, 3 * mm), figure_block(
        f_cm, 105, "圖 9：Final Improved CNN 的 validation confusion matrix。")]
    st += [figure_block(f_pc, 150, "圖 10：各方法的 per-class F1 比較。")]

    st += [H("6.1 Overfitting / Generalization 分析", 2)]
    gap_txt = []
    for k, lab, _, _ in EXPS:
        gap_txt.append("{}：train acc {} / val acc {}，gap {:+.3f}".format(
            lab.split(" ")[0], fmt(S[k]["train"]["accuracy"], 3),
            fmt(S[k]["val"]["accuracy"], 3), S[k]["generalization_gap_acc"]))
    st += bullets(gap_txt)
    cm = np.array(final["val"]["confusion_matrix"])
    off = cm.copy()
    np.fill_diagonal(off, 0)
    if off.sum() == 0:
        st += [P("Final model 在 clean validation 上的 confusion matrix 是完全"
                 "對角的（270 張全對），所以這裡<b>看不到任何錯誤案例</b>——"
                 "這本身就是 5.1 節所講的飽和現象，不代表模型沒有弱點。"
                 "真正的弱點要到劣化條件下才看得見：")]
        se = stress[final_key]
        weak = min(("lighting", "optical", "sensor_compression", "geometric"),
                   key=lambda k: se[k]["macro_f1"])
        pc = se[weak]["per_class_f1"]
        worst = int(np.argmin(pc))
        wcm = np.array(se[weak]["confusion_matrix"])
        woff = wcm.copy()
        np.fill_diagonal(woff, 0)
        i, j = np.unravel_index(woff.argmax(), woff.shape)
        st += bullets([
            "最脆弱的劣化類型是 <b>{}</b>（Macro F1 僅 {}）。".format(
                weak, fmt(se[weak]["macro_f1"])),
            "在該條件下最弱的類別是 <b>{}</b>（F1 = {}）。".format(
                CLS[worst], fmt(pc[worst])),
            "最大的錯誤流向是「真實 {} 被判成 {}」共 {} 張。".format(
                CLS[i], CLS[j], int(woff[i, j])),
        ])
        st += [P("{} 的崩潰（{}）原因很明確：Stage 1 依規定只允許基本 preprocessing，"
                 "Method A 的 augmentation 也只有幾何變換，模型從頭到尾沒看過"
                 "失焦或壓縮過的影像。這個缺口無法靠 Stage 1 的手段補，"
                 "是 Stage 2 把 blur / JPEG / 雜訊納入 augmentation 的直接動機。"
                 .format(weak, fmt(se[weak]["macro_f1"])))]
    else:
        worst = int(np.argmin(final["val"]["per_class_f1"]))
        i, j = np.unravel_index(off.argmax(), off.shape)
        st += [P("Final model 最弱的類別是 <b>{}</b>（F1 = {}）；confusion matrix "
                 "上最大的錯誤流向是「真實 {} 被判成 {}」共 {} 張。"
                 .format(CLS[worst], fmt(final["val"]["per_class_f1"][worst]),
                         CLS[i], CLS[j], int(off[i, j])))]

    if retrain:
        st += [H("6.2 最終 checkpoint 的重新訓練", 2)]
        st += [P("Model selection 完成後，依規則允許以 Train + Validation 重新訓練"
                 "一次再提交固定 checkpoint。本次最終 checkpoint 使用與 {} 完全相同的"
                 "超參數與 epoch 數（{} epochs，不再做 early stopping，因為此時已沒有"
                 "獨立的 validation），在 1,530 張影像上重新訓練。"
                 "下表中的 validation 數字僅供參考（該 split 已被納入訓練），"
                 "正式能力以 Hidden Benchmark 為準。"
                 .format(final_label, retrain["epochs_run"]))]
        st += [table([
            ["Checkpoint", "訓練資料", "Epochs", "Val Accuracy*", "Val Macro F1*"],
            [final_label + "（model selection）", "Train 1,260",
             str(final["best_epoch"]), fmt(final["val"]["accuracy"]),
             fmt(final["val"]["macro_f1"])],
            ["FINAL（提交用）", "Train + Val 1,530", str(retrain["epochs_run"]),
             fmt(retrain["val"]["accuracy"]), fmt(retrain["val"]["macro_f1"])],
        ], col_widths=[52 * mm, 36 * mm, 20 * mm, 32 * mm, 32 * mm],
            highlight_rows=(2,))]

    st += [PageBreak(), H("7. 結論", 1)]
    st += bullets([
        "傳統方法可用但明顯不足：最佳的 PCA + SVM 只有 Macro F1 {}，PCA + KNN 更只有 {}。"
        .format(fmt(sv["val"]["macro_f1"]), fmt(kn["val"]["macro_f1"])),
        "即使是 from scratch 的小型 CNN（{:,} 參數），Macro F1 已達 {}，"
        "證明局部紋理特徵對此任務是關鍵。".format(base["params"],
                                     fmt(base["val"]["macro_f1"])),
        "公開 validation 已飽和（四組 clean Macro F1 全在 {:.4f}–{:.4f} 之間），"
        "必須另外設計劣化條件下的評估，ablation 才有鑑別力。"
        .format(min(S[k]["val"]["macro_f1"] for k in S),
                max(S[k]["val"]["macro_f1"] for k in S)),
        "在 Stress 指標上，A 單獨 {:+.4f}、B 單獨 {:+.4f}，但 A+B 為 {:+.4f}；"
        "兩個方法是互補而非可加，這是四組 ablation 才看得出來的結論。"
        .format(d_a, d_b, d_ab),
        "Final Improved CNN = {}，clean Macro F1 = {}，Stress Macro F1 = {}。"
        .format(final_label, fmt(final["val"]["macro_f1"]),
                fmt(stress[final_key]["stress_macro_f1"])),
    ])
    st += [H("8. 繳交內容與執行方式", 1)]
    st += [P("python stage1_inference.py --input_dir &lt;HIDDEN_DIR&gt; "
             "--output_csv predictions.csv", "code")]
    st += bullets([
        "stage1_model.pth 為上述 Final Improved CNN 的唯一 checkpoint。",
        "inference 使用 eval mode 與 torch.no_grad()，不使用 TTA、Ensemble 或任何隨機性。",
        "preprocessing 與 validation 完全一致：灰階 → Resize(128,128) → ToTensor → Normalize。",
        "checkpoint 以相對於 stage1_inference.py 的路徑載入，不依賴 working directory。",
    ])

    RB.build(OUT, st, title="FactoryVision Stage 1 — {}".format(TEAM),
             author=TEAM)


if __name__ == "__main__":
    main()
