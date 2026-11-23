# -*- coding: utf-8 -*-
"""Build FactoryVision_Colab.ipynb — an interactive notebook for playing with
the training code (hyper-parameters, layer counts, augmentation).

It is deliberately SELF-CONTAINED: nothing is imported from Team13_Stage*/,
so editing a cell here can never affect the finished submission packages.
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "FactoryVision_Colab.ipynb")

cells = []


def md(text):
    lines = text.strip("\n").split("\n")
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]})


def code(text, title=None):
    lines = text.strip("\n").split("\n")
    meta = {}
    if title:
        meta["cellView"] = "form"
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": meta, "outputs": [],
                  "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]})


# ============================================================ 0. 封面
md(r"""
# FactoryVision — 互動式訓練實驗室

鋼材表面瑕疵辨識（NEU-DET / NEU-CLS），Team13

---

**這份 notebook 的用途**

這是一個「可以動手改」的版本，不是繳交用的。所有程式碼都**直接寫在 cell 裡**，
而不是從檔案 import，所以你可以改完一格就馬上重跑看結果。

主要能調的東西：

| 想改什麼 | 去哪一格 |
| --- | --- |
| CNN 要幾層、每層幾個通道 | 4.2 架構設定 |
| learning rate / batch size / epochs | 4.4 訓練超參數 |
| augmentation 的強度 | 4.3 資料與前處理 |
| 換架構（ResNet / DenseNet / MobileNet） | 5.1 |
| fine-tuning 策略（凍結、L2-SP…） | 5.1 |
| 偵測器的 confidence / NMS 門檻 | 6.3 |

> **重要**：這份 notebook 不會動到 `Team13_Stage1/2/3` 那三個繳交包，也不會
> 覆蓋任何已經訓練好的 checkpoint。在這裡跑出來的東西都存到 `work_colab/`。

**執行順序**：第 1、2、3 章要先跑過（環境、資料、共用工具），之後各 Stage 可以獨立跑。
""")

# ============================================================ 1. 環境
md(r"""
---
# 1. 環境設定

先確認拿到 GPU，再裝缺的套件。

**記得先切換到 GPU**：上方選單 → 執行階段 → 變更執行階段類型 → 硬體加速器選 GPU。
""")

code(r"""
# 1.1 確認硬體與 PyTorch
import subprocess, sys
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"],
                     capture_output=True, text=True).stdout or "（沒有 GPU，請切換執行階段）")

import torch, torchvision
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("CUDA 可用    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    print("顯存        : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
""")

md(r"""
### 1.2 裝套件與中文字型

Colab 已經有 torch / torchvision / sklearn / matplotlib，只缺 `ultralytics`（Stage 3 要用）。

字型這塊要特別處理：本機版用的是微軟正黑體，Linux 沒有，所以改裝 **Noto Sans CJK**。
不裝的話所有圖表的中文都會變成空白方框。
""")

code(r"""
# 1.2 套件 + 中文字型（第一次跑約 1 分鐘）
!pip install -q ultralytics
!apt-get install -qq -y fonts-noto-cjk > /dev/null 2>&1

import glob, matplotlib
from matplotlib import font_manager, rcParams

# 找一個系統裡存在的 CJK 字型（Linux 用 Noto，本機 Windows 用微軟正黑體）
CANDIDATES = sorted(glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True)) + \
             sorted(glob.glob("/usr/share/fonts/**/NotoSerifCJK*.ttc", recursive=True)) + \
             [r"C:\Windows\Fonts\msjh.ttc"]
CJK_FONT = next((p for p in CANDIDATES if os.path.exists(p)), None) if (os := __import__("os")) else None

if CJK_FONT:
    font_manager.fontManager.addfont(CJK_FONT)
    name = font_manager.FontProperties(fname=CJK_FONT).get_name()
    rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    print("中文字型:", name, "->", CJK_FONT)
else:
    print("找不到 CJK 字型，圖表中文會是方框")
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 110
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

import ultralytics
print("ultralytics:", ultralytics.__version__)
""")

code(r"""
# 1.3 快速檢查中文能不能畫出來
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 1.4))
ax.text(0.5, 0.5, "中文字型測試：鋼材表面瑕疵 crazing 0.9123",
        ha="center", va="center", fontsize=13)
ax.axis("off"); plt.show()
print("↑ 如果上面是方框，回去重跑 1.2")
""")

# ============================================================ 2. 資料
md(r"""
---
# 2. 載入資料集

整包 `STUDENT_PUBLIC` 大約 **103 MB**（classification 40 MB + detection 63 MB）。

**建議做法**：先把整個 `STUDENT_PUBLIC` 資料夾上傳到你的 Google Drive，
之後每次開 notebook 只要掛載 Drive 就好，不用重傳。

下面這一格會依序嘗試：
1. 已經掛載好的 Drive
2. Colab 本機 `/content` 底下
3. 都找不到就提示你上傳 zip
""")

code(r"""
# 2.1 設定資料集路徑
import os, glob

# ↓ 如果你的 Drive 路徑不一樣，改這裡
DRIVE_CANDIDATES = [
    "/content/drive/MyDrive/FactoryVision/STUDENT_PUBLIC",
    "/content/drive/MyDrive/STUDENT_PUBLIC",
    "/content/STUDENT_PUBLIC",
]

if not os.path.exists("/content/drive"):
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("（沒掛載 Drive，繼續找本機路徑）", e)

DATA_ROOT = next((p for p in DRIVE_CANDIDATES if os.path.isdir(p)), None)

if DATA_ROOT is None:
    print("找不到資料集。兩個選擇：")
    print("  A) 把 STUDENT_PUBLIC 上傳到 Drive 的 MyDrive/FactoryVision/ 底下，重跑這格")
    print("  B) 執行下面那格，手動上傳 STUDENT_PUBLIC.zip")
else:
    print("DATA_ROOT =", DATA_ROOT)

WORK = "/content/work_colab"      # 這份 notebook 的所有輸出都放這裡
os.makedirs(WORK, exist_ok=True)
""")

code(r"""
# 2.2 （備案）手動上傳 STUDENT_PUBLIC.zip — 用 Drive 的話跳過這格
# from google.colab import files
# up = files.upload()                      # 選 STUDENT_PUBLIC.zip
# !unzip -q -o STUDENT_PUBLIC.zip -d /content/
# DATA_ROOT = "/content/STUDENT_PUBLIC"
# print("DATA_ROOT =", DATA_ROOT)
""")

code(r"""
# 2.3 確認資料結構與張數
CLASS_ORDER = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled_in_scale", "scratches"]

cls_dir = os.path.join(DATA_ROOT, "classification")
det_dir = os.path.join(DATA_ROOT, "detection")

print("classification")
for split in ("train", "validation"):
    counts = [len(os.listdir(os.path.join(cls_dir, split, c))) for c in CLASS_ORDER]
    print("  %-11s 共 %4d 張  每類 %s" % (split, sum(counts), counts))
print("detection")
for split in ("train", "validation"):
    n_img = len(os.listdir(os.path.join(det_dir, split, "images")))
    n_xml = len(os.listdir(os.path.join(det_dir, split, "annotations_voc")))
    print("  %-11s 影像 %4d  標註 %4d" % (split, n_img, n_xml))
""")

code(r"""
# 2.4 看一下每一類長什麼樣
import matplotlib.pyplot as plt
from PIL import Image

fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for j, c in enumerate(CLASS_ORDER):
    files = sorted(os.listdir(os.path.join(cls_dir, "train", c)))[:2]
    for i, f in enumerate(files):
        ax = axes[i][j]
        ax.imshow(Image.open(os.path.join(cls_dir, "train", c, f)), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title(c, fontsize=9)
plt.tight_layout(); plt.show()
""")

# ============================================================ 3. 共用工具
md(r"""
---
# 3. 共用常數與工具

對應本機版的 `src/common.py`。這些東西所有 Stage 都會用到，所以先定義好。

`set_seed()` 一次固定四個隨機源。最後兩行 `cudnn.deterministic / benchmark`
特別重要：cuDNN 預設會自動挑最快的卷積演算法，但不同演算法的浮點累加順序不同，
結果會有微小差異。要做公平比較就得關掉。
""")

code(r"""
# 3.1 常數與工具函式
import random, json, time, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

NUM_CLASSES = len(CLASS_ORDER)
SEED = 42
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_split(split_dir):
    '''列出 (檔案路徑, 標籤)，兩層排序確保順序固定可重現。'''
    paths, labels = [], []
    for idx, cname in enumerate(CLASS_ORDER):
        cdir = os.path.join(split_dir, cname)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(IMG_EXT):
                paths.append(os.path.join(cdir, fn))
                labels.append(idx)
    return paths, labels


set_seed()
print("device:", DEVICE)
tr_p, tr_y = list_split(os.path.join(cls_dir, "train"))
va_p, va_y = list_split(os.path.join(cls_dir, "validation"))
print("train %d 張, validation %d 張" % (len(tr_p), len(va_p)))
""")

code(r"""
# 3.2 指標計算（Macro F1 等）
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

def compute_metrics(y_true, y_pred):
    labels = list(range(NUM_CLASSES))
    kw = dict(labels=labels, zero_division=0)   # 這兩個參數不能省，見下方說明
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", **kw)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, **kw).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@torch.no_grad()
def predict(model, loader):
    '''推論專用：eval() 讓 BN 改用移動平均，no_grad() 不建計算圖。'''
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        out = model(x.to(DEVICE, non_blocking=True))
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(np.asarray(y))
    return np.concatenate(preds), np.concatenate(trues)


def plot_confusion(cm, title="Confusion Matrix"):
    cm = np.array(cm, float)
    short = ["crazing", "inclusion", "patches", "pitted", "rolled", "scratches"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), short, rotation=40, ha="right")
    ax.set_yticks(range(6), short)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    ax.grid(False)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046); plt.show()

print("OK")
""")

md(r"""
> **`labels=range(6)` 和 `zero_division=0` 為什麼不能省**
>
> 規格規定「某類無有效預測時該類 F1 = 0」。不加這兩個參數的話，sklearn 遇到
> 完全沒被預測到的類別會直接**忽略**它，Macro F1 就會被高估——你看到的數字會比
> 實際好看，但 TA 跑出來不是那樣。
""")

# ============================================================ 4. Stage 1
md(r"""
---
# 4. Stage 1 — 自己從零寫的 CNN

這一章是重點。你可以改架構、改超參數、改 augmentation，然後馬上重跑看差異。
""")

md(r"""
## 4.1 先看傳統方法能做到多好

在寫 CNN 之前，先建立一個「不用深度學習」的對照組：
把影像縮到 64×64 攤平成 4096 維 → 標準化 → PCA 降維 → 分類器。

兩種分類器共用**完全相同**的前處理，只換最後一步，所以差異純粹來自分類器。
""")

code(r"""
# 4.1 PCA + SVM / KNN（約 1–2 分鐘）
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PCA_DIM   = 64      # @param {type:"integer"}
SVM_C     = 10.0    # @param {type:"number"}
SVM_GAMMA = 0.01    # @param {type:"number"}
KNN_K     = 1       # @param {type:"integer"}

FEAT = 64
def load_feats(paths):
    X = np.zeros((len(paths), FEAT * FEAT), np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L").resize((FEAT, FEAT), Image.BILINEAR)
        X[i] = np.asarray(im, np.float32).ravel() / 255.0
    return X

Xtr, Xva = load_feats(tr_p), load_feats(va_p)
ytr, yva = np.array(tr_y), np.array(va_y)

for name, clf in [("PCA + SVM (RBF)", SVC(C=SVM_C, gamma=SVM_GAMMA, kernel="rbf", random_state=SEED)),
                  ("PCA + KNN",       KNeighborsClassifier(n_neighbors=KNN_K))]:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("pca", PCA(n_components=PCA_DIM, whiten=True, random_state=SEED)),
                     ("clf", clf)])
    pipe.fit(Xtr, ytr)
    m = compute_metrics(yva, pipe.predict(Xva))
    print("%-18s Accuracy %.4f   Macro F1 %.4f" % (name, m["accuracy"], m["macro_f1"]))
""")

md(r"""
本機跑出來是 **PCA+SVM 0.7843、PCA+KNN 0.4537**。

差這麼多的原因：PCA 給的是「全域灰階強度」的線性主成分，KNN 直接在這個空間比歐式距離，
對整體亮度變化極度敏感；SVM 的 RBF kernel 至少能學非線性邊界。但兩者都**沒有任何
局部紋理的概念**——而 crazing 和 rolled_in_scale 的差別正是局部紋理的方向。

這就是接下來 CNN 能跳到 1.0 的根本原因。
""")

md(r"""
## 4.2 ★ CNN 架構設定（想改層數就改這裡）

`WIDTHS` 和 `CONVS_PER_STAGE` 這兩個 list 決定了整個網路：

- `WIDTHS = [32, 64, 128, 256]` → 四個 stage，通道數分別是 32/64/128/256
- `CONVS_PER_STAGE = [2, 2, 2, 1]` → 每個 stage 各有幾層卷積

**想加一個 stage**：兩個 list 都加一個元素，例如 `[32, 64, 128, 256, 512]` 和 `[2, 2, 2, 1, 1]`。

**注意**：每個 stage 結尾都有一次 `MaxPool2d(2)`，所以 H 和 W 會各減半。
輸入 128 經過 4 個 stage 變成 8×8；如果加到 5 個 stage 就是 4×4，加到 7 個就只剩 1×1 了。
下一格會把每一步的形狀印出來給你看。
""")

code(r"""
# 4.2 架構設定
INPUT_SIZE = 128   # @param {type:"integer"}
USE_BN     = True  # @param {type:"boolean"}
DROPOUT    = 0.0   # @param {type:"number"}
HEAD_HIDDEN = 128  # @param {type:"integer"}

WIDTHS          = [32, 64, 128, 256]   # 每個 stage 的通道數
CONVS_PER_STAGE = [2,  2,  2,   1]     # 每個 stage 有幾層卷積

assert len(WIDTHS) == len(CONVS_PER_STAGE), "兩個 list 長度必須一樣"


def conv_block(cin, cout, use_bn):
    '''Conv -> (BN) -> ReLU。開 BN 時關掉 Conv 的 bias，因為 BN 自己有 beta。'''
    layers = [nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.ReLU(inplace=True))
    return layers


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, in_ch=1, use_bn=USE_BN,
                 dropout=DROPOUT, widths=None, convs=None, head_hidden=HEAD_HIDDEN):
        super().__init__()                      # 這行不能漏
        widths = widths or WIDTHS
        convs = convs or CONVS_PER_STAGE
        feats, cin = [], in_ch
        for cout, n_conv in zip(widths, convs):
            for _ in range(n_conv):
                feats += conv_block(cin, cout, use_bn)
                cin = cout
            feats.append(nn.MaxPool2d(2))       # 每個 stage 結尾縮一半
        self.features = nn.Sequential(*feats)   # 星號 = 把 list 攤開傳進去
        self.pool = nn.AdaptiveAvgPool2d(1)     # C x H x W -> C x 1 x 1
        head = [nn.Flatten(), nn.Linear(cin, head_hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(head_hidden, num_classes))
        self.classifier = nn.Sequential(*head)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

print("架構設定完成：%d 個 stage，共 %d 層卷積" % (len(WIDTHS), sum(CONVS_PER_STAGE)))
""")

code(r"""
# 4.3 形狀檢查 — 看資料流過每一層時形狀怎麼變
model_probe = BaselineCNN()
x = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)
print("%-32s %s" % ("輸入", tuple(x.shape)))
stage = 0
for layer in model_probe.features:
    x = layer(x)
    if isinstance(layer, nn.MaxPool2d):
        stage += 1
        print("%-32s %s" % ("stage %d 之後 (MaxPool)" % stage, tuple(x.shape)))
x = model_probe.pool(x)
print("%-32s %s" % ("AdaptiveAvgPool 之後", tuple(x.shape)))
out = model_probe.classifier(x)
print("%-32s %s" % ("分類器輸出", tuple(out.shape)))

t, tr_ = count_params(model_probe)
print()
print("總參數 %s，可訓練 %s" % ("{:,}".format(t), "{:,}".format(tr_)))
flat = WIDTHS[-1] * (INPUT_SIZE // 2**len(WIDTHS))**2
print("如果不用 AdaptiveAvgPool 而直接 Flatten，光是第一層 Linear 就要 %s 個參數"
      % "{:,}".format(flat * HEAD_HIDDEN))
del model_probe
""")

md(r"""
## 4.3 資料與前處理

`eval_transform` 就是規格允許的「基本 preprocessing」三件事，
**validation 和最終推論都走這條**。

`train_transform` 多了 augmentation（Improvement A）。改 `AUG_CONFIG` 就能調強度。
""")

code(r"""
# 4.3 Dataset 與 transform
from torchvision import transforms

NORM_MEAN = [0.5070]   # 訓練集統計值（不含 validation，避免洩漏）
NORM_STD  = [0.2132]

AUG_CONFIG = {
    "hflip_p":   0.5,
    "vflip_p":   0.5,
    "rotation":  15,
    "crop_scale": (0.75, 1.0),
    "crop_ratio": (0.9, 1.111),
}


class SteelDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("L")), self.labels[i]


def eval_transform(size=INPUT_SIZE):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),                       # 同時把 0-255 縮到 0-1
        transforms.Normalize(NORM_MEAN, NORM_STD)])


def train_transform(augment, size=INPUT_SIZE):
    if not augment:
        return eval_transform(size)
    c = AUG_CONFIG
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size, scale=c["crop_scale"], ratio=c["crop_ratio"]),
        transforms.RandomHorizontalFlip(c["hflip_p"]),
        transforms.RandomVerticalFlip(c["vflip_p"]),
        transforms.RandomRotation(c["rotation"]),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)])


def make_loaders(augment, batch_size, size=INPUT_SIZE, workers=2):
    tr_ds = SteelDataset(tr_p, tr_y, train_transform(augment, size))
    va_ds = SteelDataset(va_p, va_y, eval_transform(size))
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=workers, pin_memory=True,
                       persistent_workers=workers > 0)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True,
                       persistent_workers=workers > 0)
    # 訓練集在「不含 augmentation」的前處理下再做一個 loader，用來看 overfitting
    tr_eval = DataLoader(SteelDataset(tr_p, tr_y, eval_transform(size)),
                         batch_size=batch_size, shuffle=False, num_workers=workers,
                         persistent_workers=workers > 0)
    return tr_ld, va_ld, tr_eval

print("transform 定義完成")
""")

code(r"""
# 4.3b 看看 augmentation 實際長什麼樣
set_seed(0)
tf = train_transform(True)
src = Image.open(tr_p[5]).convert("L")
fig, axes = plt.subplots(1, 7, figsize=(14, 2.4))
axes[0].imshow(src, cmap="gray"); axes[0].set_title("原圖", fontsize=9); axes[0].axis("off")
for i in range(1, 7):
    t = tf(src)[0] * NORM_STD[0] + NORM_MEAN[0]     # 反正規化才看得出來
    axes[i].imshow(t.numpy(), cmap="gray"); axes[i].axis("off")
    axes[i].set_title("aug %d" % i, fontsize=9)
plt.tight_layout(); plt.show()
""")

md(r"""
## 4.4 ★ 訓練超參數

這一格是你最常回來改的地方。改完直接跑 4.5 就會用新設定重新訓練。

> 繳交版用的是 `EPOCHS=80, BATCH=64, LR=1e-3`。
> 這裡預設 `EPOCHS=30` 是為了讓你快速試，想完整重現再調回 80。
""")

code(r"""
# 4.4 訓練超參數
EPOCHS       = 30     # @param {type:"slider", min:5, max:120, step:5}
BATCH_SIZE   = 64     # @param [16, 32, 64, 128] {type:"raw"}
LR           = 1e-3   # @param {type:"number"}
WEIGHT_DECAY = 0.0    # @param {type:"number"}
PATIENCE     = 20     # @param {type:"integer"}
AUGMENT      = True   # @param {type:"boolean"}
NUM_WORKERS  = 2      # @param {type:"integer"}

print("epochs=%d  batch=%d  lr=%g  wd=%g  augment=%s  use_bn=%s"
      % (EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, AUGMENT, USE_BN))
""")

md(r"""
## 4.5 訓練迴圈

核心就是那六個步驟。④⑤⑥ 是 PyTorch 的固定儀式，順序不能換也不能少：

- 少了 `zero_grad()` → 梯度會累加，更新量越滾越大，loss 爆炸
- 少了 `backward()` → 沒有梯度
- 少了 `step()` → 參數根本不會變
""")

code(r"""
# 4.5 訓練函式
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)            # 切換 BatchNorm / Dropout 的行為
    tot, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(x)                      # ② 前向
            loss = criterion(out, y)            # ③ 算損失
        if train:
            optimizer.zero_grad(set_to_none=True)   # ④ 清空梯度
            loss.backward()                         # ⑤ 反向
            optimizer.step()                        # ⑥ 更新
        tot += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return tot / n, correct / n


def train_model(tag="exp", augment=AUGMENT, use_bn=USE_BN, epochs=EPOCHS,
                lr=LR, batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY,
                patience=PATIENCE, dropout=DROPOUT, verbose=True):
    set_seed(SEED)
    tr_ld, va_ld, tr_eval_ld = make_loaders(augment, batch_size, workers=NUM_WORKERS)
    model = BaselineCNN(use_bn=use_bn, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_state, best_ep, bad = -1.0, None, -1, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, tr_ld, criterion, optimizer)
        vl, va = run_epoch(model, va_ld, criterion)
        pred, true = predict(model, va_ld)
        vf1 = compute_metrics(true, pred)["macro_f1"]
        scheduler.step()
        for k, v in zip(hist, (tl, ta, vl, va, vf1)):
            hist[k].append(v)
        flag = ""
        if vf1 > best_f1:
            best_f1, best_ep, bad = vf1, ep, 0
            best_state = copy.deepcopy(model.state_dict())   # deepcopy 不能省！
            flag = " *"
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or flag or ep == 1):
            print("  ep%3d  tl=%.4f ta=%.4f | vl=%.4f va=%.4f vF1=%.4f%s"
                  % (ep, tl, ta, vl, va, vf1, flag))
        if bad >= patience:
            print("  early stop @ epoch %d" % ep); break

    model.load_state_dict(best_state)
    pred, true = predict(model, va_ld)
    val_m = compute_metrics(true, pred)
    trp, trt = predict(model, tr_eval_ld)       # 用 eval transform 重算訓練集準確度
    train_m = compute_metrics(trt, trp)
    total, _ = count_params(model)
    res = {"tag": tag, "history": hist, "best_epoch": best_ep, "params": total,
           "seconds": time.time() - t0, "val": val_m, "train": train_m,
           "gap": train_m["accuracy"] - val_m["accuracy"],
           "config": {"augment": augment, "use_bn": use_bn, "epochs": epochs,
                      "lr": lr, "batch_size": batch_size, "dropout": dropout,
                      "weight_decay": weight_decay, "widths": list(WIDTHS),
                      "convs": list(CONVS_PER_STAGE)}}
    print("[%s] best ep %d | val acc=%.4f macroF1=%.4f | train acc=%.4f | gap=%+.4f | %.0f s"
          % (tag, best_ep, val_m["accuracy"], val_m["macro_f1"],
             train_m["accuracy"], res["gap"], res["seconds"]))
    return model, res

print("訓練函式就緒")
""")

md(r"""
> **為什麼訓練集準確度要用 eval transform 重算一次？**
>
> 訓練過程中的 `train_acc` 是在「加了隨機擾動」的影像上算的，會系統性偏低。
> 拿它去跟 validation 比，會誤判 overfitting 的程度。所以要用同一套前處理重算，
> `gap`（train acc − val acc）才有意義。
""")

code(r"""
# 4.6 跑一次訓練（改完 4.2 / 4.4 就回來重跑這格）
model, res = train_model(tag="my_run")
""")

code(r"""
# 4.7 畫訓練曲線 + confusion matrix
h = res["history"]
ep = range(1, len(h["train_loss"]) + 1)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].plot(ep, h["train_loss"], "--", label="train"); ax[0].plot(ep, h["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(ep, h["train_acc"], "--", label="train"); ax[1].plot(ep, h["val_acc"], label="val")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
ax[2].plot(ep, h["val_f1"], color="#b5495b")
ax[2].scatter([res["best_epoch"]], [h["val_f1"][res["best_epoch"]-1]], s=60, marker="*", zorder=5)
ax[2].set_title("Validation Macro F1（★ = 選用 epoch）"); ax[2].set_xlabel("epoch")
plt.tight_layout(); plt.show()

plot_confusion(res["val"]["confusion_matrix"],
               "val confusion matrix（Macro F1 = %.4f）" % res["val"]["macro_f1"])
print("每類 F1:", {c: round(v, 3) for c, v in zip(CLASS_ORDER, res["val"]["per_class_f1"])})
""")

md(r"""
## 4.8 四組 Ablation

兩個 improvement method 的四種組合。規格要求四組都要做，
因為只比較「Baseline vs A+B」看不出兩者是互補還是重疊。

跑完會同時印出交互作用項。
""")

code(r"""
# 4.8 跑四組 ablation（約 EPOCHS x 4 的時間）
ABL_EPOCHS = 30   # @param {type:"slider", min:5, max:120, step:5}

ablation = {}
for tag, aug, bn in [("E1_baseline", False, False),
                     ("E2_aug",      True,  False),
                     ("E3_bn",       False, True),
                     ("E4_aug_bn",   True,  True)]:
    print("=== %s (augment=%s, bn=%s) ===" % (tag, aug, bn))
    _, r = train_model(tag=tag, augment=aug, use_bn=bn, epochs=ABL_EPOCHS, verbose=False)
    ablation[tag] = r

print()
print("%-14s %-8s %-10s %-10s %-8s" % ("實驗", "A", "B", "Macro F1", "Gap"))
for tag in ("E1_baseline", "E2_aug", "E3_bn", "E4_aug_bn"):
    r = ablation[tag]
    print("%-14s %-8s %-10s %-10.4f %+.4f"
          % (tag, "O" if r["config"]["augment"] else "-",
             "O" if r["config"]["use_bn"] else "-",
             r["val"]["macro_f1"], r["gap"]))

base = ablation["E1_baseline"]["val"]["macro_f1"]
dA = ablation["E2_aug"]["val"]["macro_f1"] - base
dB = ablation["E3_bn"]["val"]["macro_f1"] - base
dAB = ablation["E4_aug_bn"]["val"]["macro_f1"] - base
print()
print("A 單獨 %+.4f, B 單獨 %+.4f, A+B %+.4f, 交互作用 %+.4f"
      % (dA, dB, dAB, dAB - dA - dB))
""")

md(r"""
## 4.9 劣化條件下的次要評估

**這一格解釋了本專案一個重要發現。**

四組實驗的 clean validation Macro F1 很可能全部擠在 0.98–1.00 之間，差距比隨機波動還小，
**根本無法選模**。原因不在模型而在資料：公開 validation 的 270 張裡有 26 張與某張訓練影像
的相似度超過 0.95，甚至有一組是位元組完全相同的檔案。

split 是課程指定不可更動的，所以解法是在**同一批** validation 影像上套用可重現的劣化，
再重新評估。注意 `np.random.default_rng(seed + i)` ——每張圖用「固定基準＋自己的索引」
建立獨立亂數源，所以第 7 張受到的劣化永遠一樣，四個模型面對的是完全相同的劣化影像。
""")

code(r"""
# 4.9 劣化評估
import io as _io
from PIL import ImageEnhance, ImageFilter

def c_lighting(img, rng):
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.45, 1.6))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.5, 1.6))
    a = np.asarray(img, np.float32) / 255.0
    return Image.fromarray((np.power(np.clip(a, 0, 1), rng.uniform(0.7, 1.5)) * 255).astype(np.uint8))

def c_optical(img, rng):
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 2.2)))
    f = rng.uniform(0.35, 0.65); w, h = img.size
    return img.resize((int(w*f), int(h*f)), Image.BILINEAR).resize((w, h), Image.BILINEAR)

def c_sensor(img, rng):
    buf = _io.BytesIO(); img.convert("L").save(buf, format="JPEG", quality=int(rng.integers(12, 35)))
    buf.seek(0); img = Image.open(buf).convert("L")
    a = np.asarray(img, np.float32) + rng.normal(0, rng.uniform(4, 14), (img.size[1], img.size[0]))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def c_geometric(img, rng):
    fill = int(np.asarray(img).mean())
    img = img.rotate(float(rng.uniform(-25, 25)), resample=Image.BILINEAR, fillcolor=fill)
    dx, dy = int(rng.integers(-18, 19)), int(rng.integers(-18, 19))
    return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy),
                         resample=Image.BILINEAR, fillcolor=fill)

CORRUPTIONS = {"lighting": c_lighting, "optical": c_optical,
               "sensor": c_sensor, "geometric": c_geometric}


class CorruptedVal(Dataset):
    def __init__(self, kind, transform, seed=777):
        self.fn, self.transform, self.seed = CORRUPTIONS[kind], transform, seed
    def __len__(self):
        return len(va_p)
    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)      # 關鍵：每張圖固定劣化
        return self.transform(self.fn(Image.open(va_p[i]).convert("L"), rng)), va_y[i]


def stress_eval(model, batch_size=64):
    tf = eval_transform()
    out = {}
    for kind in CORRUPTIONS:
        ld = DataLoader(CorruptedVal(kind, tf), batch_size=batch_size, shuffle=False)
        p, t = predict(model, ld)
        out[kind] = compute_metrics(t, p)["macro_f1"]
    out["stress_mean"] = float(np.mean([out[k] for k in CORRUPTIONS]))
    return out

# 看一眼劣化後長什麼樣
fig, axes = plt.subplots(1, 5, figsize=(13, 2.6))
axes[0].imshow(Image.open(va_p[7]).convert("L"), cmap="gray")
axes[0].set_title("原圖", fontsize=9); axes[0].axis("off")
for i, (kind, fn) in enumerate(CORRUPTIONS.items(), 1):
    axes[i].imshow(fn(Image.open(va_p[7]).convert("L"), np.random.default_rng(777+7)), cmap="gray")
    axes[i].set_title(kind, fontsize=9); axes[i].axis("off")
plt.tight_layout(); plt.show()

s = stress_eval(model)
print("目前模型:", {k: round(v, 4) for k, v in s.items()})
""")

code(r"""
# 4.9b 把四組 ablation 都做劣化評估（要先跑過 4.8）
if "ablation" in dir() and ablation:
    print("%-14s %-9s %-9s %-9s %-9s %-9s %-9s"
          % ("實驗", "Clean", "lighting", "optical", "sensor", "geometric", "Stress均"))
    for tag in ("E1_baseline", "E2_aug", "E3_bn", "E4_aug_bn"):
        cfg = ablation[tag]["config"]
        set_seed(SEED)
        m, _ = train_model(tag=tag, augment=cfg["augment"], use_bn=cfg["use_bn"],
                           epochs=cfg["epochs"], verbose=False)
        s = stress_eval(m)
        print("%-14s %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f"
              % (tag, ablation[tag]["val"]["macro_f1"], s["lighting"], s["optical"],
                 s["sensor"], s["geometric"], s["stress_mean"]))
        del m; torch.cuda.empty_cache()
else:
    print("請先跑 4.8")
""")

# ============================================================ 5. Stage 2
md(r"""
---
# 5. Stage 2 — 遷移學習

改用 ImageNet 預訓練的 backbone。重點在兩件事：

1. **怎麼換掉分類頭** — ImageNet 輸出 1000 類，我們只要 6 類
2. **怎麼凍結參數** — 其實只是 `p.requires_grad = False` 一行
""")

code(r"""
# 5.1 架構與 fine-tuning 策略
from torchvision import models

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]
S2_INPUT   = 224


def build_backbone(arch, pretrained=True):
    '''回傳 (模型, 分類頭的屬性名稱)。不同架構的頭叫法不一樣。'''
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        return m, "fc"
    if arch == "densenet121":
        w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.densenet121(weights=w)
        m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
        return m, "classifier"
    if arch == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
        return m, "classifier"
    raise ValueError(arch)


def apply_strategy(model, head_attr, strategy):
    if strategy == "feature_extraction":
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith(head_attr)   # 只有頭可以訓練
    else:                                               # full_finetune / l2sp
        for p in model.parameters():
            p.requires_grad = True
    return model


class L2SP:
    '''L2-SP：全層訓練，但把 backbone 拉住在 ImageNet 的解附近。
       loss 多一項 alpha/2 * ||w - w0||^2（backbone）+ beta/2 * ||w||^2（新頭）'''
    def __init__(self, model, head_attr, alpha=0.01, beta=0.01):
        self.alpha, self.beta = alpha, beta
        # detach 切斷計算圖，clone 複製記憶體（不 clone 的話會跟著被更新）
        self.ref = {n: p.detach().clone() for n, p in model.named_parameters()
                    if not n.startswith(head_attr)}
    def to(self, device):
        self.ref = {n: v.to(device) for n, v in self.ref.items()}
        return self
    def __call__(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in self.ref:
                loss = loss + self.alpha * 0.5 * ((p - self.ref[n]) ** 2).sum()
            else:
                loss = loss + self.beta * 0.5 * (p ** 2).sum()
        return loss

# 看一下三種架構、三種策略的可訓練參數比例
for arch in ("resnet18", "densenet121", "mobilenet_v3_small"):
    for st in ("feature_extraction", "full_finetune"):
        m, head = build_backbone(arch, pretrained=False)
        apply_strategy(m, head, st)
        t, tr_ = count_params(m)
        print("%-20s %-20s total=%10s trainable=%10s (%.2f%%)"
              % (arch, st, "{:,}".format(t), "{:,}".format(tr_), 100*tr_/t))
        del m
""")

md(r"""
## 5.2 對應 Robustness 的 augmentation

Stage 2 的 benchmark 會用**同一份 checkpoint** 跑四次：正常 + lighting + optical +
sensor_compression，而且禁止依資料夾名稱切換行為。

既然推論時不能做任何事，唯一的辦法就是訓練時讓模型見過這些劣化。
""")

code(r"""
# 5.2 Stage 2 的 augmentation
import io as _io2

class RandomGamma:
    def __init__(self, rng=(0.6, 1.6), p=0.3): self.rng, self.p = rng, p
    def __call__(self, img):
        if random.random() < self.p:
            img = transforms.functional.adjust_gamma(img, random.uniform(*self.rng))
        return img

class RandomDownUp:
    def __init__(self, factor=(0.4, 0.9), p=0.2): self.factor, self.p = factor, p
    def __call__(self, img):
        if random.random() < self.p:
            f = random.uniform(*self.factor); w, h = img.size
            img = img.resize((max(8,int(w*f)), max(8,int(h*f))), Image.BILINEAR).resize((w,h), Image.BILINEAR)
        return img

class RandomJPEG:
    '''torchvision 沒有內建，自己寫一個：存成 JPEG 再讀回來就有壓縮失真。'''
    def __init__(self, quality=(25, 85), p=0.3): self.quality, self.p = quality, p
    def __call__(self, img):
        if random.random() < self.p:
            buf = _io2.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=random.randint(*self.quality))
            buf.seek(0); img = Image.open(buf).convert(img.mode)
        return img

class RandomNoise:
    def __init__(self, std=(0.01, 0.07), p=0.3): self.std, self.p = std, p
    def __call__(self, t):
        if random.random() < self.p:
            t = (t + torch.randn_like(t) * random.uniform(*self.std)).clamp(0, 1)
        return t


class SteelRGB(Dataset):
    '''灰階複製成 3 通道，因為 ImageNet backbone 要 3 通道輸入。'''
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("L").convert("RGB")), self.labels[i]


def s2_eval_tf(size=S2_INPUT):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(IMNET_MEAN, IMNET_STD)])

def s2_train_tf(size=S2_INPUT, robust=True):
    geo = [transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
           transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
           transforms.RandomRotation(15)]
    if not robust:
        return transforms.Compose(geo + [transforms.ToTensor(),
                                         transforms.Normalize(IMNET_MEAN, IMNET_STD)])
    photo = [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4)], p=0.8),  # lighting
             RandomGamma(),                                                      # lighting
             transforms.RandomApply([transforms.GaussianBlur(5, (0.3, 2.0))], p=0.3),  # optical
             RandomDownUp(),                                                     # optical
             RandomJPEG()]                                                       # compression
    tail = [transforms.ToTensor(), RandomNoise(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))]
    return transforms.Compose(geo + photo + tail)

print("Stage 2 transform 就緒")
""")

code(r"""
# 5.3 ★ Stage 2 超參數
S2_ARCH     = "resnet18"  # @param ["resnet18", "densenet121", "mobilenet_v3_small"]
S2_STRATEGY = "l2sp"      # @param ["feature_extraction", "full_finetune", "l2sp"]
S2_EPOCHS   = 12          # @param {type:"slider", min:3, max:40, step:1}
S2_BATCH    = 32          # @param [8, 16, 24, 32, 48] {type:"raw"}
S2_LR       = 1e-4        # @param {type:"number"}
S2_WD       = 1e-4        # @param {type:"number"}
S2_LABEL_SMOOTH = 0.05    # @param {type:"number"}
L2SP_ALPHA  = 0.01        # @param {type:"number"}
L2SP_BETA   = 0.01        # @param {type:"number"}
S2_ROBUST_AUG = True      # @param {type:"boolean"}

print("%s + %s, %d epochs, batch %d, lr %g" % (S2_ARCH, S2_STRATEGY, S2_EPOCHS, S2_BATCH, S2_LR))
""")

code(r"""
# 5.4 訓練 Stage 2
def train_stage2(arch=S2_ARCH, strategy=S2_STRATEGY, epochs=S2_EPOCHS,
                 batch_size=S2_BATCH, lr=S2_LR, wd=S2_WD, robust=S2_ROBUST_AUG,
                 amp=True, verbose=True):
    set_seed(SEED)
    tr_ld = DataLoader(SteelRGB(tr_p, tr_y, s2_train_tf(robust=robust)),
                       batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                       pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    va_ld = DataLoader(SteelRGB(va_p, va_y, s2_eval_tf()), batch_size=batch_size,
                       shuffle=False, num_workers=NUM_WORKERS,
                       persistent_workers=NUM_WORKERS > 0)

    model, head = build_backbone(arch, pretrained=True)
    apply_strategy(model, head, strategy)
    model.to(DEVICE)
    reg = L2SP(model, head, L2SP_ALPHA, L2SP_BETA).to(DEVICE) if strategy == "l2sp" else None
    params = [p for p in model.parameters() if p.requires_grad]   # 只傳可訓練的！
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if reg else wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=S2_LABEL_SMOOTH)
    scaler = torch.amp.GradScaler("cuda") if (amp and DEVICE.type == "cuda") else None

    total, trainable = count_params(model)
    print("%s + %s  可訓練 %s / %s (%.2f%%)"
          % (arch, strategy, "{:,}".format(trainable), "{:,}".format(total), 100*trainable/total))

    best_f1, best_state, best_ep = -1.0, None, -1
    hist = {"train_loss": [], "val_acc": [], "val_f1": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot = n = 0
        for x, y in tr_ld:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            if scaler:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x); loss = criterion(out, y)
                if reg: loss = loss + reg(model)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                out = model(x); loss = criterion(out, y)
                if reg: loss = loss + reg(model)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        scheduler.step()
        pred, true = predict(model, va_ld)
        m = compute_metrics(true, pred)
        hist["train_loss"].append(tot/n); hist["val_acc"].append(m["accuracy"])
        hist["val_f1"].append(m["macro_f1"])
        flag = ""
        if m["macro_f1"] > best_f1:
            best_f1, best_ep = m["macro_f1"], ep
            best_state = copy.deepcopy({k: v.detach().cpu().clone()
                                        for k, v in model.state_dict().items()})
            flag = " *"
        if verbose:
            print("  ep%3d loss=%.4f | val acc=%.4f F1=%.4f%s"
                  % (ep, tot/n, m["accuracy"], m["macro_f1"], flag))
    model.load_state_dict(best_state); model.to(DEVICE)
    pred, true = predict(model, va_ld)
    val_m = compute_metrics(true, pred)
    print("[%s+%s] best ep %d | val F1 %.4f | %.0f s"
          % (arch, strategy, best_ep, val_m["macro_f1"], time.time()-t0))
    return model, {"val": val_m, "history": hist, "best_epoch": best_ep,
                   "params": total, "trainable": trainable}

s2_model, s2_res = train_stage2()
""")

code(r"""
# 5.5 Stage 2 的 robustness 探測
class CorruptedRGB(Dataset):
    def __init__(self, kind, transform, seed=1234):
        self.fn, self.transform, self.seed = CORRUPTIONS[kind], transform, seed
    def __len__(self): return len(va_p)
    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        img = self.fn(Image.open(va_p[i]).convert("L"), rng).convert("RGB")
        return self.transform(img), va_y[i]

tf = s2_eval_tf()
rob = {}
for kind in ("lighting", "optical", "sensor"):
    ld = DataLoader(CorruptedRGB(kind, tf), batch_size=S2_BATCH, shuffle=False)
    p, t = predict(s2_model, ld)
    rob[kind] = compute_metrics(t, p)["macro_f1"]
rob["robust_mean"] = float(np.mean(list(rob.values())))

print("clean        %.4f" % s2_res["val"]["macro_f1"])
for k, v in rob.items():
    print("%-12s %.4f" % (k, v))
plot_confusion(s2_res["val"]["confusion_matrix"], "Stage 2 %s + %s" % (S2_ARCH, S2_STRATEGY))
""")

# ============================================================ 6. Stage 3
md(r"""
---
# 6. Stage 3 — 物件偵測

用 Ultralytics 的 YOLOv8s。這一章比較特別的是**自己實作 mAP@0.5**——
因為各家框架的 mAP 定義細節不一樣，而正式成績以 TA 的定義為準。

> Colab 的 T4 有 16 GB 顯存，可以用 batch 16 甚至 32（本機 4 GB 只能用 8）。
""")

code(r"""
# 6.1 把公開 split 轉成 Ultralytics 要的資料夾結構
import shutil, xml.etree.ElementTree as ET

DS = os.path.join(WORK, "det_dataset")

def prepare_yolo_dataset():
    for split, folder in (("train", "train"), ("val", "validation")):
        src_img = os.path.join(det_dir, folder, "images")
        src_lbl = os.path.join(det_dir, folder, "labels_yolo")
        dst_img = os.path.join(DS, "images", split)
        dst_lbl = os.path.join(DS, "labels", split)
        os.makedirs(dst_img, exist_ok=True); os.makedirs(dst_lbl, exist_ok=True)
        for fn in sorted(os.listdir(src_img)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in IMG_EXT: continue
            shutil.copyfile(os.path.join(src_img, fn), os.path.join(dst_img, fn))
            lbl = os.path.join(src_lbl, stem + ".txt")
            dst = os.path.join(dst_lbl, stem + ".txt")
            if os.path.exists(lbl):
                shutil.copyfile(lbl, dst)
            else:
                open(dst, "w").close()      # 沒有物件的影像也要給一個空檔
    yaml_path = os.path.join(DS, "factoryvision.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: %s\n" % DS)
        f.write("train: images/train\nval: images/val\n\nnames:\n")
        for i, c in enumerate(CLASS_ORDER):
            f.write("  %d: %s\n" % (i, c))
    return yaml_path

YAML = prepare_yolo_dataset()
print("train %d, val %d" % (len(os.listdir(os.path.join(DS, "images", "train"))),
                            len(os.listdir(os.path.join(DS, "images", "val")))))
print("yaml:", YAML)
""")

code(r"""
# 6.2 ★ Stage 3 訓練參數
DET_MODEL  = "yolov8s.pt"  # @param ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
DET_EPOCHS = 60            # @param {type:"slider", min:10, max:200, step:10}
DET_BATCH  = 16            # @param [8, 16, 24, 32] {type:"raw"}
DET_IMGSZ  = 640           # @param [320, 416, 512, 640] {type:"raw"}
DET_LR0    = 0.01          # @param {type:"number"}
DET_DEGREES  = 10.0        # @param {type:"number"}
DET_SCALE    = 0.5         # @param {type:"number"}
DET_MOSAIC   = 1.0         # @param {type:"number"}
DET_CLOSE_MOSAIC = 20      # @param {type:"integer"}

print("%s  %d epochs  batch %d  imgsz %d" % (DET_MODEL, DET_EPOCHS, DET_BATCH, DET_IMGSZ))
""")

code(r"""
# 6.3 訓練 YOLO（T4 上 60 epochs 約 30-40 分鐘）
from ultralytics import YOLO

det = YOLO(DET_MODEL)
results = det.train(
    data=YAML, epochs=DET_EPOCHS, imgsz=DET_IMGSZ, batch=DET_BATCH,
    lr0=DET_LR0, optimizer="auto", seed=SEED, deterministic=True,
    degrees=DET_DEGREES, translate=0.1, scale=DET_SCALE,
    fliplr=0.5, flipud=0.5, mosaic=DET_MOSAIC, close_mosaic=DET_CLOSE_MOSAIC,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.4,
    project=os.path.join(WORK, "stage3_runs"), name="colab", exist_ok=True,
    workers=2, plots=True,
)
BEST = os.path.join(str(results.save_dir), "weights", "best.pt")
print("best weights:", BEST)
""")

md(r"""
## 6.4 自己實作 mAP@0.5

完全照題目定義：

1. 每個類別，把所有預測依 confidence 由高到低排序
2. 逐一比對 Ground Truth，取 IoU 最大的那個
3. IoU ≥ 0.5、類別正確、**且該 GT 還沒被認領** → TP，否則 FP
4. 重複框、類別錯、IoU 不足 → 一律 FP
5. 每類 AP 用 Pascal VOC 2010+ all-points 插值
6. 六類 AP 不加權平均

`used` 那個布林陣列就是在實作「一個 GT 最多只能匹配一個 prediction」。
""")

code(r"""
# 6.4 mAP 實作
def parse_voc(xml_path):
    ALIAS = {"rolled-in_scale": "rolled_in_scale"}   # VOC 用連字號，輸出要底線
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        name = ALIAS.get(name, name)
        if name not in CLASS_ORDER: continue
        bb = obj.find("bndbox")
        boxes.append((CLASS_ORDER.index(name),
                      float(bb.find("xmin").text), float(bb.find("ymin").text),
                      float(bb.find("xmax").text), float(bb.find("ymax").text)))
    return w, h, boxes


def load_gt(xml_dir):
    gt = {}
    for fn in sorted(os.listdir(xml_dir)):
        if not fn.endswith(".xml"): continue
        w, h, boxes = parse_voc(os.path.join(xml_dir, fn))
        gt[os.path.splitext(fn)[0]] = {"size": (w, h), "boxes": boxes}
    return gt


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    area_a = (a[:,2]-a[:,0])*(a[:,3]-a[:,1]); area_b = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return (inter / np.maximum(area_a[:,None] + area_b[None,:] - inter, 1e-12)).astype(np.float32)


def voc_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre)-2, -1, -1):          # 由右往左做成單調遞減階梯
        mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))


def evaluate_detections(gt, preds, iou_thr=0.5, score_thr=0.25):
    per_class, aps = {}, []
    for c in range(NUM_CLASSES):
        gt_by_img, n_gt = {}, 0
        for iid, rec in gt.items():
            boxes = [b[1:] for b in rec["boxes"] if b[0] == c]
            if boxes:
                gt_by_img[iid] = {"boxes": np.array(boxes, np.float64),
                                  "used": np.zeros(len(boxes), bool)}
                n_gt += len(boxes)
        rows = sorted([(iid, p[1], p[2:]) for iid, pl in preds.items() for p in pl if p[0] == c],
                      key=lambda r: -r[1])
        tp, fp = np.zeros(len(rows)), np.zeros(len(rows))
        for i, (iid, _s, box) in enumerate(rows):
            rec = gt_by_img.get(iid)
            if rec is None: fp[i] = 1; continue
            ious = iou_matrix(np.array([box]), rec["boxes"])[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not rec["used"][j]:
                rec["used"][j] = True; tp[i] = 1
            else:
                fp[i] = 1
        if n_gt == 0:
            ap = 0.0
        else:
            ctp, cfp = np.cumsum(tp), np.cumsum(fp)
            ap = voc_ap(ctp/n_gt, ctp/np.maximum(ctp+cfp, 1e-12))
        keep = [i for i, r in enumerate(rows) if r[1] >= score_thr]
        tp_t, fp_t = float(tp[keep].sum()), float(fp[keep].sum())
        per_class[CLASS_ORDER[c]] = {"ap50": ap, "n_gt": n_gt,
                                     "precision": tp_t/max(tp_t+fp_t, 1e-12),
                                     "recall": tp_t/max(n_gt, 1e-12)}
        aps.append(ap)
    return {"map50": float(np.mean(aps)), "per_class": per_class}

print("mAP 實作完成")
""")

code(r"""
# 6.5 評估 + 門檻掃描
CONF = 0.001   # @param {type:"number"}
NMS  = 0.6     # @param {type:"number"}
MAXD = 100     # @param {type:"integer"}

gt = load_gt(os.path.join(det_dir, "validation", "annotations_voc"))
val_imgs = sorted(glob.glob(os.path.join(det_dir, "validation", "images", "*.png")))
best_model = YOLO(BEST)

def run_preds(conf, nms, maxd):
    preds = {}
    for i in range(0, len(val_imgs), 16):
        chunk = val_imgs[i:i+16]
        for p, r in zip(chunk, best_model.predict(chunk, imgsz=DET_IMGSZ, conf=conf,
                                                  iou=nms, max_det=maxd, verbose=False)):
            iid = os.path.splitext(os.path.basename(p))[0]
            out = []
            if r.boxes is not None and len(r.boxes):
                xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                out = [(int(cl[j]), float(cf[j]), *[float(v) for v in xy[j]]) for j in range(len(cf))]
            preds[iid] = out
    return preds

preds = run_preds(CONF, NMS, MAXD)
res3 = evaluate_detections(gt, preds)
print("mAP@0.5 = %.4f" % res3["map50"])
for c, v in res3["per_class"].items():
    print("  %-16s AP=%.4f  GT=%3d  P=%.3f R=%.3f" % (c, v["ap50"], v["n_gt"], v["precision"], v["recall"]))

print()
print("門檻掃描（說明為什麼提交時要用很低的 conf）")
for cf in (0.001, 0.005, 0.01, 0.05, 0.25):
    p = run_preds(cf, NMS, MAXD)
    n = np.mean([len(v) for v in p.values()])
    print("  conf=%-6s mAP=%.4f  平均 %.1f 框/張" % (cf, evaluate_detections(gt, p)["map50"], n))
""")

code(r"""
# 6.6 GT vs Prediction 視覺化
import matplotlib.patches as mpatches
SHOW_CONF = 0.25   # @param {type:"number"}

ids = sorted(gt)[:8]
fig, axes = plt.subplots(2, len(ids), figsize=(2.2*len(ids), 5.0))
for k, iid in enumerate(ids):
    img = np.asarray(Image.open(os.path.join(det_dir, "validation", "images", iid + ".png")).convert("RGB"))
    for row, (boxes, color, lab) in enumerate([
            ([(b[0], None, *b[1:]) for b in gt[iid]["boxes"]], "#00c853", "GT"),
            ([d for d in preds.get(iid, []) if d[1] >= SHOW_CONF], "#ff1744", "Pred")]):
        ax = axes[row][k]; ax.imshow(img); ax.axis("off")
        ax.set_title("%s %s" % (lab, iid), fontsize=7)
        for b in boxes:
            x1, y1, x2, y2 = b[2:]
            ax.add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                            edgecolor=color, lw=1.6))
            txt = CLASS_ORDER[b[0]] + ("" if b[1] is None else " %.2f" % b[1])
            ax.text(x1+1, max(y1-3, 7), txt, color="white", fontsize=5.5,
                    bbox=dict(facecolor=color, edgecolor="none", pad=0.6, alpha=0.85))
plt.tight_layout(); plt.show()
""")

# ============================================================ 7. 存檔
md(r"""
---
# 7. 把結果存起來

Colab session 一斷，`/content` 底下的東西全部消失。想保留就存到 Drive。

> **再次提醒**：這份 notebook 跑出來的 checkpoint **不是**繳交用的。
> 正式繳交的是本機那三個 `Team13_StageN.zip`，裡面的 checkpoint 已經跑過完整實驗、
> 也通過了格式驗證。這裡的東西只是給你實驗用的。
""")

code(r"""
# 7.1 存到 Drive
SAVE_DIR = "/content/drive/MyDrive/FactoryVision/colab_outputs"  # @param {type:"string"}
os.makedirs(SAVE_DIR, exist_ok=True)

# Stage 1
if "model" in dir():
    torch.save({"state_dict": model.state_dict(), "config": res["config"]},
               os.path.join(SAVE_DIR, "stage1_colab.pth"))
    json.dump({k: v for k, v in res.items() if k != "history"},
              open(os.path.join(SAVE_DIR, "stage1_colab.json"), "w"), indent=2)
    print("存了 stage1_colab.pth")

# Stage 2
if "s2_model" in dir():
    torch.save({"state_dict": s2_model.state_dict(),
                "config": {"architecture": S2_ARCH, "strategy": S2_STRATEGY,
                           "input_size": S2_INPUT, "norm_mean": IMNET_MEAN,
                           "norm_std": IMNET_STD, "class_order": CLASS_ORDER}},
               os.path.join(SAVE_DIR, "stage2_colab.pth"))
    print("存了 stage2_colab.pth")

# Stage 3
if "BEST" in dir() and os.path.exists(BEST):
    shutil.copyfile(BEST, os.path.join(SAVE_DIR, "stage3_colab.pt"))
    print("存了 stage3_colab.pt")

print("全部存到:", SAVE_DIR)
""")

md(r"""
---
## 附：想比較不同設定時的小技巧

跑多組設定時，把結果收進一個 list，最後一起比較：

```python
runs = []
for w in ([16, 32, 64, 128], [32, 64, 128, 256], [64, 128, 256, 512]):
    WIDTHS = w
    _, r = train_model(tag="w=%s" % w[0], epochs=20, verbose=False)
    runs.append(r)

for r in runs:
    print("%-10s params=%9s  Macro F1=%.4f" %
          (r["tag"], "{:,}".format(r["params"]), r["val"]["macro_f1"]))
```

幾個值得試的方向：

- **把 `WIDTHS` 整體縮小一半** → 參數量剩約 1/4，看 Macro F1 掉多少
- **加第五個 stage** → `WIDTHS=[32,64,128,256,512]`、`CONVS_PER_STAGE=[2,2,2,1,1]`，
  注意 128 經過 5 次 pooling 只剩 4×4
- **`DROPOUT` 調到 0.3 ~ 0.5** → 看 gap（train acc − val acc）有沒有縮小
- **`LR` 調成 1e-2 或 1e-4** → 看曲線震盪或收斂變慢
- **`AUGMENT=False` 搭配 `USE_BN=False`** → 退回最原始的 baseline
""")

# ============================================================ 輸出
nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True,
                  "name": "FactoryVision_Colab.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with io.open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print("wrote %s" % OUT)
print("  markdown %d cells, code %d cells, 共 %d" % (n_md, n_code, len(cells)))
print("  %.1f KB" % (os.path.getsize(OUT) / 1024))
