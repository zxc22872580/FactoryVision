# -*- coding: utf-8 -*-
"""Diagrams drawn specially for the code-walkthrough Word document."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams

CJK = r"C:\Windows\Fonts\msjh.ttc"
if os.path.exists(CJK):
    font_manager.fontManager.addfont(CJK)
    rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "doc")
os.makedirs(OUT, exist_ok=True)

BLUE = "#3b6ea5"
ORANGE = "#e07b39"
GREEN = "#4f9d69"
RED = "#b5495b"
PURPLE = "#7a5ba6"
GREY = "#8a8a8a"
LIGHT = "#eef2f7"
NL = chr(10)   # newline helper for multi-line labels


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("->", p)
    return p


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=9, bold=False, tc="#111"):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                   linewidth=1.4, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, text="", color=GREY, fs=8, rad=0.0, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                linestyle=ls,
                                connectionstyle="arc3,rad=%.2f" % rad))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, text, ha="center",
                va="bottom", fontsize=fs, color=color)


# ---------------------------------------------------------------- 1. overview
def fig_overview():
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.01, 0.56, 0.17, 0.30,
        "FactoryVision\nPublic Dataset\n\nTrain 1,260\nVal 270",
        fc="#fff7e6", ec=ORANGE, bold=True)
    box(ax, 0.235, 0.70, 0.22, 0.24,
        "Stage 1\n自建 CNN（from scratch）\n這是什麼瑕疵？", fc="#e8f0f8")
    box(ax, 0.235, 0.40, 0.22, 0.24,
        "Stage 2\n遷移學習（ImageNet）\n這是什麼瑕疵？", fc="#e8f0f8")
    box(ax, 0.235, 0.08, 0.22, 0.24,
        "Stage 3\n物件偵測（YOLOv8s）\n瑕疵在哪裡？", fc="#e8f0f8")
    for y in (0.82, 0.52, 0.20):
        arrow(ax, 0.185, 0.71, 0.232, y)
    box(ax, 0.52, 0.70, 0.17, 0.24, "stage1_model.pth\n615,974 參數",
        fc="#eaf5ee", ec=GREEN)
    box(ax, 0.52, 0.40, 0.17, 0.24, "stage2_model.pth\n11.2M 參數",
        fc="#eaf5ee", ec=GREEN)
    box(ax, 0.52, 0.08, 0.17, 0.24, "stage3_model.pt\n11.1M 參數",
        fc="#eaf5ee", ec=GREEN)
    for y in (0.82, 0.52, 0.20):
        arrow(ax, 0.457, y, 0.517, y, "訓練")
    box(ax, 0.755, 0.70, 0.225, 0.24,
        "stage1_inference.py\n→ predictions.csv", fc="#fdeef0", ec=RED)
    box(ax, 0.755, 0.40, 0.225, 0.24,
        "stage2_inference.py\n→ predictions.csv ×4", fc="#fdeef0", ec=RED)
    box(ax, 0.755, 0.08, 0.225, 0.24,
        "stage3_detect.py\n→ detections.json", fc="#fdeef0", ec=RED)
    for y in (0.82, 0.52, 0.20):
        arrow(ax, 0.692, y, 0.752, y, "載入")
    ax.text(0.5, 0.985, "三個 Stage 的整體架構", ha="center", fontsize=13,
            fontweight="bold")
    ax.text(0.868, 0.015, "← TA 只會執行這一欄", ha="center", fontsize=8.5,
            color=RED)
    return save(fig, "overview.png")


# ------------------------------------------------------------ 2. file graph
def fig_file_graph():
    """Clean layered dependency graph: arrows only go downwards."""
    fig, ax = plt.subplots(figsize=(11, 7.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.985, "Stage 1 的檔案相依關係（箭頭 = import / 使用）",
            ha="center", fontsize=13, fontweight="bold")

    for y, lab in [(0.875, "第 1 層@共用常數"),
                   (0.655, "第 2 層@可重複使用的零件"),
                   (0.425, "第 3 層@實際跑實驗的程式"),
                   (0.140, "第 4 層@產物與繳交")]:
        ax.text(0.012, y, lab.replace("@", NL), fontsize=8.5, color=GREY,
                va="center", linespacing=1.5)

    box(ax, 0.30, 0.825, 0.42, 0.105,
        "common.py@類別順序・seed・正規化常數・列檔工具".replace("@", NL),
        fc="#fff7e6", ec=ORANGE, bold=True)

    box(ax, 0.17, 0.600, 0.20, 0.110,
        "model.py@定義 CNN 架構".replace("@", NL), fc="#e8f0f8")
    box(ax, 0.41, 0.600, 0.20, 0.110,
        "dataset.py@讀圖 + transform".replace("@", NL), fc="#e8f0f8")
    box(ax, 0.65, 0.600, 0.20, 0.110,
        "evaluate.py@指標・延遲量測".replace("@", NL), fc="#e8f0f8")
    for x in (0.27, 0.51, 0.75):
        arrow(ax, 0.51, 0.822, x, 0.714)

    box(ax, 0.10, 0.370, 0.235, 0.110,
        "train.py@訓練主程式（E1–E4）".replace("@", NL),
        fc="#eaf5ee", ec=GREEN, bold=True)
    box(ax, 0.375, 0.370, 0.235, 0.110,
        "stress_eval.py@劣化條件下的次要評估".replace("@", NL),
        fc="#eaf5ee", ec=GREEN)
    box(ax, 0.655, 0.370, 0.235, 0.110,
        "traditional_baseline.py@PCA + SVM / KNN".replace("@", NL),
        fc="#eaf5ee", ec=GREEN)

    arrow(ax, 0.24, 0.597, 0.20, 0.484)
    arrow(ax, 0.47, 0.597, 0.25, 0.484)
    arrow(ax, 0.71, 0.597, 0.29, 0.484)
    arrow(ax, 0.31, 0.597, 0.45, 0.484)
    arrow(ax, 0.75, 0.597, 0.53, 0.484)
    arrow(ax, 0.69, 0.828, 0.80, 0.484, rad=-0.18, ls="--")

    box(ax, 0.055, 0.080, 0.27, 0.115,
        "work/stage1_runs/@E1–E4 的 .pth 與 _summary.json".replace("@", NL),
        fc="#f2eef7", ec=PURPLE)
    arrow(ax, 0.20, 0.367, 0.19, 0.198, "輸出")

    box(ax, 0.375, 0.080, 0.235, 0.115,
        "stage1_model.pth@唯一提交的 checkpoint".replace("@", NL),
        fc="#fdeef0", ec=RED, bold=True)
    arrow(ax, 0.327, 0.138, 0.372, 0.138, "挑最佳")

    box(ax, 0.655, 0.080, 0.27, 0.115,
        "stage1_inference.py@TA 唯一執行的檔案".replace("@", NL),
        fc="#fdeef0", ec=RED, bold=True)
    arrow(ax, 0.612, 0.138, 0.652, 0.138, "載入")
    arrow(ax, 0.79, 0.597, 0.82, 0.198, rad=0.22, ls="--")
    ax.text(0.952, 0.40,
            "也 import@common.py@與 model.py".replace("@", NL), fontsize=7.6,
            color=GREY, ha="center", linespacing=1.5)
    return save(fig, "file_graph.png")


# --------------------------------------------------------- 3. tensor shapes
def fig_tensor_flow():
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(-0.012, 1.012); ax.set_ylim(0, 1); ax.axis("off")
    stages = [
        ("輸入影像@(灰階)", "1 x 128 x 128", "#fff7e6", ORANGE),
        ("stage1@Conv x2 + Pool", "32 x 64 x 64", "#e8f0f8", BLUE),
        ("stage2@Conv x2 + Pool", "64 x 32 x 32", "#e8f0f8", BLUE),
        ("stage3@Conv x2 + Pool", "128 x 16 x 16", "#e8f0f8", BLUE),
        ("stage4@Conv x1 + Pool", "256 x 8 x 8", "#e8f0f8", BLUE),
        ("AdaptiveAvgPool@+ Flatten", "256", "#eaf5ee", GREEN),
        ("Linear(256->128)@ReLU@Linear(128->6)", "6", "#fdeef0", RED),
    ]
    n = len(stages)
    gap = 0.018
    w = (1.0 - gap * (n - 1)) / n
    for i, (name, shape, fc, ec) in enumerate(stages):
        x = i * (w + gap)
        box(ax, x, 0.46, w, 0.30, name.replace("@", NL), fc=fc, ec=ec, fs=8.0)
        ax.text(x + w / 2, 0.37, shape, ha="center", va="center", fontsize=8.8,
                fontweight="bold", color="#333")
        ax.text(x + w / 2, 0.295,
                "(C x H x W)" if i <= 4 else "(一維向量)",
                ha="center", fontsize=7, color=GREY)
        if i < n - 1:
            arrow(ax, x + w + 0.001, 0.61, x + w + gap - 0.001, 0.61)
    ax.text(0.5, 0.95, "張量形狀怎麼一路變化", ha="center", fontsize=13,
            fontweight="bold")
    ax.text(0.5, 0.865,
            "每經過一次 MaxPool2d(2)，高和寬各減半；每次進入新 stage，"
            "通道數加倍。空間資訊逐步換成語意資訊。",
            ha="center", fontsize=9, color="#444")
    ax.text(0.5, 0.15,
            "關鍵在倒數第二步：256x8x8 = 16,384 個數字，"
            "若直接 Flatten 接 Linear(16384->128) 會產生 2,097,152 個參數；@"
            "改用 AdaptiveAvgPool 先把每個 channel 壓成 1 個數字，"
            "只剩 Linear(256->128) 的 32,768 個參數，省了 98%。".replace("@", NL),
            ha="center", fontsize=8.8, color=RED, linespacing=1.7)
    return save(fig, "tensor_flow.png")


# ------------------------------------------------------------- 4. padding
def fig_padding():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, pad in zip(axes, (0, 1)):
        n = 5
        size = n + 2 * pad
        grid = np.zeros((size, size))
        if pad:
            grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 0.35
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=1)
        for i in range(size + 1):
            ax.axhline(i - 0.5, color="#999", lw=0.6)
            ax.axvline(i - 0.5, color="#999", lw=0.6)
        ax.add_patch(mp.Rectangle((-0.5, -0.5), 3, 3, fill=False,
                                  edgecolor=RED, lw=2.4))
        out = size - 2
        ax.set_title(("padding=0：5×5 → %d×%d（縮小了）" % (out, out)) if not pad
                     else ("padding=1：5×5 → %d×%d（尺寸不變）" % (out, out)),
                     fontsize=10.5, fontweight="bold",
                     color=RED if not pad else GREEN)
        ax.set_xticks([]); ax.set_yticks([])
        if pad:
            ax.text(size / 2 - 0.5, size + 0.15,
                    "淺色格 = 補上的 0", ha="center", fontsize=8.5, color=GREY)
    fig.suptitle("為什麼 nn.Conv2d(..., kernel_size=3, padding=1)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.06,
             "紅框是 3×3 的卷積核。沒有 padding 時，核心無法對準最邊緣的像素，"
             "輸出就比輸入小 2 格；\n每層都縮 2 格的話，7 層卷積後 128 會變成 114，"
             "尺寸難以計算。padding=1 補一圈 0，輸出尺寸就和輸入一樣。",
             ha="center", fontsize=8.8, color="#444", linespacing=1.6)
    fig.tight_layout()
    return save(fig, "padding.png")


# -------------------------------------------------------- 5. training loop
def fig_train_loop():
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.30, 0.90, 0.40, 0.075, "for epoch in range(1, 81):",
        fc="#fff7e6", ec=ORANGE, bold=True, fs=10)
    steps = [
        ("① for x, y in train_loader:", "DataLoader 每次給一批 64 張影像\n"
         "（此時才真的讀檔、做 augmentation）", 0.775, BLUE),
        ("② out = model(x)", "前向傳播：影像 → 6 個分數\n"
         "（這一步會呼叫 forward()）", 0.645, BLUE),
        ("③ loss = criterion(out, y)", "算出「錯多少」的單一數字", 0.515, BLUE),
        ("④ optimizer.zero_grad()", "把上一批殘留的梯度清空\n"
         "（PyTorch 預設是累加，不清會錯）", 0.385, RED),
        ("⑤ loss.backward()", "反向傳播：算出每個參數該往哪動", 0.255, RED),
        ("⑥ optimizer.step()", "依梯度實際更新參數", 0.125, RED),
    ]
    for code, desc, y, c in steps:
        box(ax, 0.045, y, 0.33, 0.095, code, fc="#f3f4f6", ec=c, fs=9.2,
            bold=True)
        ax.text(0.40, y + 0.047, desc, va="center", fontsize=8.8,
                color="#333", linespacing=1.5)
    for i in range(len(steps) - 1):
        arrow(ax, 0.21, steps[i][2], 0.21, steps[i + 1][2] + 0.095)
    arrow(ax, 0.50, 0.885, 0.21, 0.872)
    ax.annotate("", xy=(0.045, 0.82), xytext=(0.015, 0.17),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6,
                                connectionstyle="arc3,rad=0.42"))
    ax.text(0.012, 0.50, "下一批", rotation=90, ha="right", va="center",
            fontsize=8.5, color=GREEN)
    ax.text(0.5, 0.99, "一個 epoch 裡到底發生什麼", ha="center", fontsize=13,
            fontweight="bold")
    ax.text(0.5, 0.045,
            "④⑤⑥ 這三步是 PyTorch 的固定儀式，順序不能換、也不能少。"
            "少了 ④ 梯度會累加導致更新量爆炸；少了 ⑤ 沒有梯度；少了 ⑥ 參數根本不會變。",
            ha="center", fontsize=8.8, color="#444", linespacing=1.6)
    return save(fig, "train_loop.png")


# ------------------------------------------------- 6. train vs inference path
def fig_two_paths():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(mp.FancyBboxPatch((0.01, 0.52), 0.98, 0.42,
                                   boxstyle="round,pad=0.01", lw=1.2,
                                   edgecolor=BLUE, facecolor="#f7fafd"))
    ax.add_patch(mp.FancyBboxPatch((0.01, 0.04), 0.98, 0.42,
                                   boxstyle="round,pad=0.01", lw=1.2,
                                   edgecolor=RED, facecolor="#fef8f9"))
    ax.text(0.025, 0.90, "訓練時（我們自己跑）", fontsize=11,
            fontweight="bold", color=BLUE)
    ax.text(0.025, 0.42, "Benchmark 時（TA 跑）", fontsize=11,
            fontweight="bold", color=RED)
    tr = ["train/ 資料夾\n每類一個子目錄", "train_transform\n含 augmentation",
          "model.train()\n梯度開啟", "optimizer\n更新參數",
          "存成 .pth\nstate_dict + config"]
    inf = ["一個扁平資料夾\n只有影像檔", "eval_transform\nResize/ToTensor/Normalize",
           "model.eval()\ntorch.no_grad()", "argmax\n取最高分類別",
           "寫出 predictions.csv\nimage_id,prediction"]
    for i, t in enumerate(tr):
        box(ax, 0.03 + i * 0.194, 0.58, 0.168, 0.22, t, fc="white", ec=BLUE,
            fs=8.2)
        if i < 4:
            arrow(ax, 0.198 + i * 0.194, 0.69, 0.224 + i * 0.194, 0.69, "",
                  BLUE)
    for i, t in enumerate(inf):
        box(ax, 0.03 + i * 0.194, 0.10, 0.168, 0.22, t, fc="white", ec=RED,
            fs=8.2)
        if i < 4:
            arrow(ax, 0.198 + i * 0.194, 0.21, 0.224 + i * 0.194, 0.21, "",
                  RED)
    arrow(ax, 0.945, 0.575, 0.945, 0.325, "", GREEN, ls="--")
    ax.text(0.90, 0.45, "同一份\nweights", fontsize=8, color=GREEN,
            ha="center")
    ax.text(0.5, 0.985,
            "為什麼訓練程式和推論程式要分開寫", ha="center", fontsize=13,
            fontweight="bold")
    return save(fig, "two_paths.png")


# ----------------------------------------------------------- 7. freeze graph
def fig_freeze():
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    names = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc\n(新的)"]
    for row, (title, frozen, c) in enumerate([
            ("Feature Extraction：只訓練最後的分類層", 5, ORANGE),
            ("Full Fine-tuning：全部一起訓練", 0, GREEN)]):
        y = 0.56 - row * 0.40
        ax.text(0.02, y + 0.26, title, fontsize=10.5, fontweight="bold",
                color=c)
        for i, n in enumerate(names):
            is_frozen = i < frozen
            box(ax, 0.02 + i * 0.158, y, 0.145, 0.20, n,
                fc="#e5e7eb" if is_frozen else "#eaf5ee",
                ec=GREY if is_frozen else GREEN, fs=8.5,
                tc=GREY if is_frozen else "#111")
            ax.text(0.0925 + i * 0.158, y - 0.065,
                    "凍結" if is_frozen else "可訓練", ha="center", fontsize=7.6,
                    color=GREY if is_frozen else GREEN)
        ax.text(0.98, y + 0.10, "requires_grad\n= %s" % ("False / True" if frozen
                                                         else "True"),
                ha="right", fontsize=8, color="#444")
    ax.text(0.5, 0.97, "兩種 fine-tuning 策略在程式碼裡的差別", ha="center",
            fontsize=13, fontweight="bold")
    ax.text(0.5, 0.015,
            "程式碼上只是一行 p.requires_grad = False —— 被凍結的參數不會產生梯度，"
            "optimizer 也就不會更新它。",
            ha="center", fontsize=8.8, color="#444")
    return save(fig, "freeze.png")


# ------------------------------------------------------------- 8. map steps
def fig_map():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [
        ("① 收集這一類的所有預測框", "把 270 張影像中、類別 = crazing 的框全部放在一起"),
        ("② 依 confidence 由高到低排序", "分數最高的框先被檢查"),
        ("③ 逐一比對 Ground Truth", "算 IoU，取最大的那個 GT"),
        ("④ 判定 TP 還是 FP",
         "IoU ≧ 0.5 且該 GT 還沒被配對過 → TP；否則 → FP\n"
         "（重複框、類別錯、IoU 不足，一律 FP）"),
        ("⑤ 算累積 Precision / Recall", "cumsum(TP) / n_gt 與 cumsum(TP) / (TP+FP)"),
        ("⑥ VOC 2010+ all-points 積分", "先把 precision 由右往左取最大值（做成階梯），再算曲線下面積 = AP"),
        ("⑦ 六類 AP 取平均", "不加權平均 = mAP@0.5"),
    ]
    for i, (t, d) in enumerate(steps):
        y = 0.86 - i * 0.123
        box(ax, 0.02, y, 0.30, 0.095, t, fc="#f3f4f6", ec=BLUE, fs=8.8,
            bold=True)
        ax.text(0.345, y + 0.047, d, va="center", fontsize=8.5, color="#333",
                linespacing=1.5)
        if i < len(steps) - 1:
            arrow(ax, 0.17, y, 0.17, y - 0.028)
    ax.text(0.5, 0.985, "自己實作的 mAP@0.5：完全照題目定義的七個步驟",
            ha="center", fontsize=13, fontweight="bold")
    return save(fig, "map_steps.png")


if __name__ == "__main__":
    fig_overview()
    fig_file_graph()
    fig_tensor_flow()
    fig_padding()
    fig_train_loop()
    fig_two_paths()
    fig_freeze()
    fig_map()
    print("all diagrams written to", OUT)
