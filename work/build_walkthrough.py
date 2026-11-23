# -*- coding: utf-8 -*-
"""Build FactoryVision_程式碼完整講解.docx."""
import os
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from docx_kit import (BLUE, CJK, Doc, GREY, NAVY, RED, STEEL,  # noqa: E402
                      set_font)

FIG = os.path.join(HERE, "figures")
OUT = os.path.join(os.path.dirname(HERE), "FactoryVision_程式碼完整講解.docx")


def F(name):
    return os.path.join(FIG, name)


def main():
    d = Document()
    for s in d.sections:
        s.top_margin = Inches(0.9)
        s.bottom_margin = Inches(0.9)
        s.left_margin = Inches(0.9)
        s.right_margin = Inches(0.9)
    style = d.styles["Normal"]
    style.font.name = CJK
    style.font.size = Pt(10.5)
    D = Doc(d)

    # ---------------------------------------------------------- cover
    for _ in range(6):
        d.add_paragraph()
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("FactoryVision Challenge"), CJK, 26, True, NAVY)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("程式碼完整講解"), CJK, 22, True, STEEL)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    set_font(p.add_run("從「我知道 CNN 有卷積層和池化層」"), CJK, 11, False, GREY)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("到「我看得懂這份程式碼每一行在做什麼」"), CJK, 11,
             False, GREY)
    for _ in range(4):
        d.add_paragraph()
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("鋼材表面瑕疵辨識與定位系統"), CJK, 10, False, GREY)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("Stage 1 – 3 ＋ Final Report"), CJK, 10, False, GREY)

    # ======================================================= 讀法
    D.h("這份文件怎麼讀", 1, page_break=True)
    D.para("這份文件假設你已經學過卷積、池化、activation、loss 這些概念，但不確定它們「翻成程式碼長什麼樣」。所以它刻意不重複講理論，而是把重點放在三件事：")
    D.bullet([{"t": "概念怎麼變成程式碼。", "b": True},
              {"t": "你知道「卷積層後面要接 activation」，但不知道為什麼程式裡寫 "},
              {"t": "nn.Conv2d(...)", "mono": True},
              {"t": " 時要加 "}, {"t": "padding=1", "mono": True},
              {"t": "，也不知道 "}, {"t": "bias=not use_bn", "mono": True},
              {"t": " 是什麼意思。"}])
    D.bullet([{"t": "為什麼要有這個檔案。", "b": True},
              {"t": "整個專案三十幾個檔案，每一個的存在都有理由——通常是為了讓「我們自己訓練」和「繳交後被 TA 執行」這兩條路徑能共用同一份定義，又不互相干擾。"}])
    D.bullet([{"t": "為什麼要有這一行。", "b": True},
              {"t": "有些行看起來可有可無，其實拿掉就會出錯、或讓實驗失去公平性。這類地方會特別標出來。"}])
    D.para("閱讀順序：第 2 到第 5 章是基礎，看完之後第 6 到第 8 章的逐檔講解會變得很好懂。只想查某個檔案的話，可以直接跳到對應章節。")
    D.note("符號約定",
           "灰底等寬字是程式碼。黃框是「這裡寫錯會出事」的提醒。每張圖都附說明，圖可以單獨看懂。")

    # ======================================================= 第 2 章
    D.h("第 2 章　先看懂全局", 1, page_break=True)

    D.h("2.1　三個 Stage 在做什麼", 2)
    D.para("整個專題是一條逐步加深的線。Stage 1 用自己從零寫的 CNN 回答「這張影像是哪一種瑕疵」；Stage 2 改用在 ImageNet 上預訓練好的模型做同一件事，比較哪種遷移策略好；Stage 3 把問題升級成「瑕疵在哪裡」，要輸出框的座標。")
    D.figure(F("doc/overview.png"),
             "圖 2-1　三個 Stage 的整體架構。最右邊那一欄是 TA 真正會執行的程式，其餘都是我們自己跑的。")
    D.para("請特別注意圖最右邊那一欄，這是整個專案架構的關鍵：TA 跑分時只會執行那三支腳本中的一支，而且只給一個裝滿影像的資料夾。他不會跑我們的訓練程式、不會看 notebook、也不會幫我們改路徑。整個檔案結構都是繞著這個限制設計的。")

    D.h("2.2　訓練與推論是兩條不同的路", 2)
    D.para("初學者最常見的誤解是「訓練完直接拿同一段程式碼來預測就好」。實際上這兩條路徑的需求完全不同，硬要共用會出問題。")
    D.figure(F("doc/two_paths.png"),
             "圖 2-2　訓練路徑（上）與推論路徑（下）。唯一共享的是那份權重檔。")
    D.para("差別在三個地方：")
    D.bullet([{"t": "資料結構不同。", "b": True},
              {"t": "訓練時是 "}, {"t": "train/crazing/、train/inclusion/…", "mono": True},
              {"t": " 這種「每類一個子資料夾」的結構，資料夾名稱就是標籤；推論時 TA 給的是扁平資料夾，裡面全是檔名像 "},
              {"t": "HID_0001.png", "mono": True}, {"t": " 的影像，沒有任何標籤。"}])
    D.bullet([{"t": "前處理不同。", "b": True},
              {"t": "訓練時做隨機翻轉、旋轉來增加資料多樣性；推論時絕對不能有任何隨機性，否則同一張圖跑兩次會得到不同答案。"}])
    D.bullet([{"t": "模型狀態不同。", "b": True},
              {"t": "訓練時要算梯度、要更新參數；推論時兩者都不要，而且 BatchNorm 的行為必須切換（第 5 章詳述）。"}])
    D.para("所以專案裡會有 train.py 和 stage1_inference.py 兩支程式。它們共用模型架構的定義與常數，但各走各的流程。")

    D.h("2.3　為什麼不把所有東西寫在一個檔案裡", 2)
    D.para("一個 500 行的 main.py 當然也能跑完，但這個專案有三個硬性理由必須拆開：")
    D.bullet([{"t": "規格強制要求。", "b": True},
              {"t": "繳交規格明寫 src/ 底下必須有 model.py、train.py、evaluate.py，而且「只交 inference code 不算完成實作」。"}])
    D.bullet([{"t": "推論程式不能依賴訓練程式。", "b": True},
              {"t": "TA 只安裝 requirements.txt 然後跑一行指令。如果推論腳本要 import train.py，而 train.py 又 import 了只有訓練時才裝的套件，benchmark 會直接掛掉。"}])
    D.bullet([{"t": "公平比較需要單一真相來源。", "b": True},
              {"t": "Stage 1 要跑四組 ablation。如果四組各有一份程式碼，很容易不小心讓某一組的 learning rate 不一樣，整個比較就失去意義。寫成同一支程式加兩個開關，差異就只可能來自那兩個開關。"}])

    # ======================================================= 第 3 章
    D.h("第 3 章　檔案地圖", 1, page_break=True)

    D.h("3.1　每個繳交包裡的七件固定檔案", 2)
    D.para("這七個檔名是規格硬性規定的，檔名錯就算格式錯誤，benchmark 直接視為執行失敗。")
    D.table(["檔案", "用處", "誰會用到"], [
        ["stageN_report.pdf", "該階段的評分證據。規格明寫「只有 code、沒有指定結果或圖表，該項目不計分」", "TA 人工評分"],
        ["stageN_inference.py@@stage3_detect.py", "唯一的 benchmark 入口，TA 只會跑這一行", "TA 自動跑分"],
        ["stageN_model.pth / .pt", "唯一的 checkpoint，不准放多個讓程式自己挑", "被上面那支載入"],
        ["config.json", "把模型設定攤開成機器可讀格式，規格列了必填欄位", "TA 核對報告與程式是否一致"],
        ["requirements.txt", "安裝環境用，禁止 URL、本機路徑、editable install", "TA 裝環境"],
        ["README.md", "Python 版本、模型名稱、preprocessing、TA 該跑的指令", "TA 閱讀"],
        ["src/ 的三個檔案", "model.py、train.py、evaluate.py，證明有完整訓練流程", "TA 人工檢視"],
    ], [150, 330, 130])

    D.h("3.2　Stage 1 的檔案相依關係", 2)
    D.para("下面這張圖是整份文件最值得先看懂的一張。它把 Stage 1 的七個程式檔分成四層，箭頭代表「誰 import 誰」，而且箭頭一律由上往下，沒有循環相依。")
    D.figure(F("doc/file_graph.png"),
             "圖 3-1　Stage 1 的檔案相依關係。第 1 層是所有人都要用的常數，第 4 層是最終產物。")
    D.para("這個分層有一個簡單的判斷原則：越上層的檔案，越「不知道自己會被誰用」。common.py 完全不知道 train.py 的存在，它只負責提供類別順序和正規化常數；model.py 只負責定義架構，不管誰來訓練它。")
    D.para("好處在後來才顯現：當我們需要加 stress_eval.py 做額外評估時，它可以直接重用 model.py 和 evaluate.py，完全不必改動任何既有檔案。")

    D.h("3.3　為什麼是這五個檔案", 2)
    D.table(["檔案", "負責什麼", "沒有它會怎樣"], [
        ["common.py", "類別順序、random seed、正規化常數、列檔工具", "類別順序可能在訓練和推論時不一致，每張預測都對應到錯的類別"],
        ["model.py", "CNN 架構的定義", "推論時無法重建出和訓練時一樣的網路，權重載不進去"],
        ["dataset.py", "怎麼讀圖、怎麼做前處理", "訓練和驗證的前處理可能不一致，指標失真"],
        ["train.py", "訓練迴圈、early stopping、存檔", "沒有訓練流程，規格直接不給分"],
        ["evaluate.py", "指標計算、confusion matrix、延遲量測", "每個地方各算各的指標，數字對不起來"],
    ], [110, 190, 300])
    D.note("關鍵觀念",
           "這五個檔案的分法在 Stage 2 和 Stage 3 被原封不動沿用。看懂一次，三個 Stage 都看得懂。")

    # ======================================================= 第 4 章
    D.h("第 4 章　從概念到程式碼：CNN 怎麼寫出來", 1, page_break=True)
    D.para("這一章是整份文件的核心。我們用 Stage 1 的 Baseline CNN 當例子，把「腦中的 CNN」一步步翻成 PyTorch 程式碼。")

    D.h("4.1　你腦中的 CNN vs 程式裡的 nn.Module", 2)
    D.para("你腦中的 CNN 大概長這樣：影像進來 → 卷積 → activation → 池化 → 再卷積 → …… → 最後接一個分類器輸出六個類別的分數。在 PyTorch 裡，這件事被拆成兩個步驟：")
    D.bullet([{"t": "__init__", "mono": True, "b": True},
              {"t": "：宣告零件。把需要的每一層建立出來，存成物件的屬性。這一步只是準備零件，資料還沒開始流動。"}])
    D.bullet([{"t": "forward", "mono": True, "b": True},
              {"t": "：規定資料怎麼流過這些零件。真正的計算順序寫在這裡。"}])
    D.code([
        "class BaselineCNN(nn.Module):",
        "    def __init__(self, num_classes=6, in_ch=1, use_bn=False, ...):",
        "        super().__init__()",
        "        self.features = nn.Sequential(...)      # 卷積的部分",
        "        self.pool = nn.AdaptiveAvgPool2d(1)     # 全域平均池化",
        "        self.classifier = nn.Sequential(...)    # 分類器",
        "",
        "    def forward(self, x):",
        "        return self.classifier(self.pool(self.features(x)))",
    ], "model.py 的骨架")
    D.para([{"t": "為什麼要繼承 "}, {"t": "nn.Module", "mono": True},
            {"t": "？因為 PyTorch 靠它來自動追蹤「這個模型有哪些可學習的參數」。你把一層存成 "},
            {"t": "self.xxx", "mono": True},
            {"t": "，PyTorch 就會自動把那層的權重註冊進來，之後 "},
            {"t": "model.parameters()", "mono": True},
            {"t": " 才拿得到它們交給 optimizer。如果你把層存進一個普通的 Python list，PyTorch 看不到，那些參數永遠不會被訓練。"}])
    D.note("新手最常踩的坑",
           "忘記寫 super().__init__()。少了這一行，nn.Module 的內部結構沒有初始化，之後所有註冊參數的動作都會失敗。")

    D.h("4.2　nn.Conv2d 的參數到底在說什麼", 2)
    D.code([
        "nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,",
        "          padding=1, bias=False)",
    ], "一層卷積的完整寫法")
    D.table(["參數", "意思", "我們為什麼這樣設"], [
        ["in_channels", "輸入有幾個通道", "第一層是 1，因為鋼材影像是灰階。彩色就是 3"],
        ["out_channels", "要產生幾張特徵圖，也就是這層有幾個卷積核", "32→64→128→256，越深層越多，用通道數換取空間解析度"],
        ["kernel_size", "卷積核的邊長", "3。兩個 3×3 疊起來的感受野等於一個 5×5，但參數更少"],
        ["padding", "在影像四周補幾圈 0", "1。配合 kernel_size=3 可讓輸出尺寸和輸入一樣大"],
        ["bias", "要不要額外的偏移項", "有 BatchNorm 時設 False，因為 BN 自己就有偏移項"],
    ], [100, 175, 320])
    D.para("padding 是最容易困惑的一個。下面這張圖解釋為什麼 padding=1 能讓尺寸不變：")
    D.figure(F("doc/padding.png"),
             "圖 4-1　padding 的作用。沒有 padding 時卷積核碰不到最邊緣的像素，輸出就會縮小。", 5.6)
    D.para("如果每層都縮小 2 格，七層卷積後 128 會變成 114，而且每次改架構都要重算尺寸。加了 padding=1 之後，尺寸只在 MaxPool 時改變，心算就能追蹤。")

    D.h("4.3　張量形狀怎麼一路變化", 2)
    D.para("「張量」聽起來很玄，其實就是多維陣列。在 CNN 裡，一批影像的形狀是 (N, C, H, W)：N 是這批有幾張、C 是通道數、H 和 W 是高和寬。")
    D.figure(F("doc/tensor_flow.png"),
             "圖 4-2　資料流過 Baseline CNN 時的形狀變化。", 6.3)
    D.para("看懂這張圖，你就能自己推算任何 CNN 的輸出尺寸。規則只有兩條：")
    D.bullet([{"t": "Conv2d(kernel=3, padding=1)", "mono": True},
              {"t": " 改變通道數 C，不改變 H 和 W。"}])
    D.bullet([{"t": "MaxPool2d(2)", "mono": True},
              {"t": " 不改變通道數，把 H 和 W 各除以 2。"}])

    D.h("4.4　為什麼是 Conv → BN → ReLU 這個順序", 2)
    D.code([
        "def conv_block(cin, cout, use_bn):",
        "    layers = [nn.Conv2d(cin, cout, kernel_size=3, padding=1,",
        "                        bias=not use_bn)]",
        "    if use_bn:",
        "        layers.append(nn.BatchNorm2d(cout))",
        "    layers.append(nn.ReLU(inplace=True))",
        "    return layers",
    ], "model.py：一個可重複使用的卷積區塊")
    D.para("這個小函式做了三件值得說明的事。")

    D.h("（1）為什麼回傳 list 而不是 nn.Sequential", 3)
    D.para([{"t": "因為回傳 list 之後可以用 "}, {"t": "+", "mono": True},
            {"t": " 串接，最後再一次包成 Sequential："}])
    D.code([
        "feats = []",
        "feats += conv_block(in_ch, c1, use_bn) + conv_block(c1, c1, use_bn)",
        "feats += [nn.MaxPool2d(2)]                 # 128 -> 64",
        "feats += conv_block(c1, c2, use_bn) + conv_block(c2, c2, use_bn)",
        "feats += [nn.MaxPool2d(2)]                 # 64 -> 32",
        "...",
        "self.features = nn.Sequential(*feats)",
    ])
    D.para([{"t": "最後那個 "}, {"t": "*feats", "mono": True},
            {"t": " 是 Python 的「解包」語法："}, {"t": "nn.Sequential", "mono": True},
            {"t": " 要的是一個一個傳進去的參數，不是一個 list，所以用星號把 list 攤開。這是很多人第一次看 PyTorch 程式碼時會卡住的地方。"}])

    D.h("（2）bias=not use_bn 在做什麼", 3)
    D.para("BatchNorm 的公式是 γ·(x−μ)/σ + β，那個 β 已經是一個可學習的偏移。如果前面的 Conv 再帶自己的 bias，減去平均值 μ 的時候 bias 會被一起減掉，等於完全沒作用卻佔著參數。所以開 BN 就關 Conv 的 bias——這也是 ResNet 原始實作的做法。")

    D.h("（3）順序為什麼是 Conv → BN → ReLU", 3)
    D.para("BN 要放在 activation 之前，才能把進入 ReLU 的數值分佈拉回零均值、單位變異數。如果放在 ReLU 之後，輸入已經被砍掉一半（負的都變 0），分佈嚴重偏斜，正規化的效果就大打折扣。")
    D.note("這個設計同時服務了 ablation 實驗",
           "use_bn 是一個布林參數，E1/E2（不開 BN）和 E3/E4（開 BN）走的是同一段程式碼。這樣就能確定兩組的差異真的只來自 BN，而不是因為寫了兩份程式碼而不小心有別的差異。")

    D.h("4.5　MaxPool 與 AdaptiveAvgPool 的分工", 2)
    D.para("網路裡有兩種池化，用途完全不同。")
    D.bullet([{"t": "MaxPool2d(2)", "mono": True, "b": True},
              {"t": "：在每個 2×2 小區塊裡取最大值，把特徵圖縮小一半。目的是擴大感受野、減少計算量。"}])
    D.bullet([{"t": "AdaptiveAvgPool2d(1)", "mono": True, "b": True},
              {"t": "：不管輸入多大，都把每個通道壓成一個數字（整張特徵圖的平均值）。目的是把空間維度徹底消掉，接上分類器。"}])
    D.para("第二個是整個架構裡最划算的設計決定。如果不用它，而是直接 Flatten：")
    D.code([
        "# 作法 A：直接攤平（不推薦）",
        "nn.Flatten()                 # 256 x 8 x 8 -> 16384",
        "nn.Linear(16384, 128)        # 參數 = 16384 x 128 = 2,097,152",
        "",
        "# 作法 B：先全域平均池化（我們用的）",
        "nn.AdaptiveAvgPool2d(1)      # 256 x 8 x 8 -> 256 x 1 x 1",
        "nn.Flatten()                 # -> 256",
        "nn.Linear(256, 128)          # 參數 = 256 x 128 = 32,768",
    ])
    D.para("差了 64 倍。而且作法 B 還有額外好處：全域平均讓模型對「瑕疵出現在畫面哪個位置」不敏感，正好符合任務特性——刮痕在左上角還是右下角，都還是刮痕。")

    D.h("4.6　權重初始化為什麼不能用預設值", 2)
    D.code([
        "for m in self.modules():",
        "    if isinstance(m, nn.Conv2d):",
        "        nn.init.kaiming_normal_(m.weight, mode='fan_out',",
        "                                nonlinearity='relu')",
        "    elif isinstance(m, nn.BatchNorm2d):",
        "        nn.init.ones_(m.weight)",
        "        nn.init.zeros_(m.bias)",
    ], "model.py：_init_weights()")
    D.para([{"t": "ReLU 會把一半的輸出變成 0，等於讓訊號的變異數減半。七層疊下來訊號會嚴重衰減，深層幾乎收不到有效梯度。He 初始化（"},
            {"t": "kaiming_normal_", "mono": True},
            {"t": "）用 √(2/fan_out) 這個係數把損失補回來，讓訊號在各層之間維持穩定。"}])
    D.para("另外那兩行 BN 的初始化：weight 設 1、bias 設 0，意思是「一開始先什麼都不做」——BN 剛開始等同於單純的標準化，讓模型自己在訓練中學出需要的 γ 和 β。")

    D.h("4.7　forward() 是誰呼叫的", 2)
    D.para([{"t": "你在程式裡從來不會直接寫 "}, {"t": "model.forward(x)", "mono": True},
            {"t": "，而是寫 "}, {"t": "model(x)", "mono": True},
            {"t": "。因為 nn.Module 定義了 __call__，它會在呼叫 forward 之前和之後做一些額外的事（例如觸發 hook）。直接呼叫 forward 會跳過那些機制，是公認的錯誤用法。"}])
    D.code([
        "out = model(x)            # 正確：會經過 nn.Module.__call__",
        "out = model.forward(x)    # 不要這樣寫",
    ])

    # ======================================================= 第 5 章
    D.h("第 5 章　訓練迴圈：那幾行程式碼背後", 1, page_break=True)
    D.para("幾乎所有 PyTorch 訓練程式都長得一模一樣。看懂這一章，你就看懂了專案裡所有的 train.py。")
    D.figure(F("doc/train_loop.png"),
             "圖 5-1　一個 epoch 裡實際發生的六個步驟。", 5.8)

    D.h("5.1　為什麼一定要 zero_grad", 2)
    D.para([{"t": "PyTorch 的梯度是「累加」的，不是「覆寫」的。如果你連續跑兩次 "},
            {"t": "loss.backward()", "mono": True},
            {"t": " 而中間沒有清空，第二次算出來的梯度會加到第一次上面。"}])
    D.para("這個設計是刻意的——它讓「用小 batch 模擬大 batch」這種技巧變得可能。但代價是你必須每一批都手動清空，忘記的話參數更新量會越滾越大，loss 直接爆炸。")
    D.code([
        "optimizer.zero_grad(set_to_none=True)",
        "loss.backward()",
        "optimizer.step()",
    ], "train.py 裡的三連擊")
    D.para([{"t": "那個 "}, {"t": "set_to_none=True", "mono": True},
            {"t": " 是小優化：把梯度設成 None 比填 0 省記憶體也稍快，因為不需要真的去寫一塊全 0 的張量。"}])

    D.h("5.2　model.train() 和 model.eval() 差在哪", 2)
    D.para("這兩個方法不會改變任何權重，它們只切換某些層的「行為模式」。受影響的主要是兩種層：")
    D.table(["層", "train() 模式", "eval() 模式"], [
        ["BatchNorm", "用「當前這一批」的平均值和變異數來標準化，同時更新內部的移動平均", "用訓練期間累積的移動平均，不再更新"],
        ["Dropout", "隨機丟棄一部分神經元", "完全不丟棄，全部保留"],
    ], [90, 250, 240])
    D.note("為什麼這件事在這個專案特別重要",
           "規格明文禁止「隨機 inference」。如果推論時忘了呼叫 eval()，BatchNorm 會改用當前 batch 的統計量，那麼同一張影像和不同的影像湊成一批，就會得到不同的預測結果。這不只是準確度問題，而是直接違規。")
    D.code([
        "def run_epoch(model, loader, device, criterion, optimizer=None):",
        "    train = optimizer is not None",
        "    model.train(train)   # 傳 True 是 train 模式，False 是 eval",
    ], "train.py：一個函式同時服務訓練與驗證")
    D.para("這裡用了一個小技巧：用「有沒有傳 optimizer 進來」來判斷現在是訓練還是驗證。這樣訓練和驗證就共用同一段迴圈，不會出現「驗證那段忘了跟著改」的問題。")

    D.h("5.3　torch.no_grad() 省的是什麼", 2)
    D.para("PyTorch 在前向傳播的同時會建立一張「計算圖」，記錄每一步運算，之後反向傳播才知道怎麼回推梯度。這張圖很佔記憶體。")
    D.para([{"t": "推論時根本不需要梯度，所以用 "}, {"t": "@torch.no_grad()", "mono": True},
            {"t": " 把整個函式包起來，告訴 PyTorch 不要建圖。效果是記憶體用量大幅下降、速度變快，而且避免不小心修改到參數。"}])
    D.code([
        "@torch.no_grad()",
        "def predict(model, loader, device):",
        "    model.eval()",
        "    for x, y in loader:",
        "        logit = model(x.to(device))",
        "        preds.append(logit.argmax(1).cpu().numpy())",
    ], "evaluate.py：推論專用函式")
    D.para([{"t": "最後那行 "}, {"t": "logit.argmax(1)", "mono": True},
            {"t": " 是把六個分數中最大的那個的「索引」取出來。參數 1 代表沿著第 1 個維度（類別維度）取，因為第 0 個維度是 batch。"}])

    D.h("5.4　Early Stopping 與一個容易踩的陷阱", 2)
    D.code([
        "if vf1 > best_f1:",
        "    best_f1, best_epoch, bad = vf1, ep, 0",
        "    best_state = copy.deepcopy(model.state_dict())   # 注意這裡",
        "else:",
        "    bad += 1",
        "if bad >= args.patience:",
        "    break",
    ], "train.py：early stopping")
    D.note("這裡的 deepcopy 不能省",
           "state_dict() 回傳的是「參照」，不是複本。如果直接寫 best_state = model.state_dict()，模型繼續訓練時會把那份資料一起改掉，最後存出去的根本不是最佳權重，而是最後一個 epoch 的權重。這個 bug 很隱蔽，因為程式不會報錯，只會讓成績莫名其妙變差。")
    D.para([{"t": "另外注意我們監控的是 "}, {"t": "validation Macro F1", "b": True},
            {"t": " 而不是 loss。因為評分指標就是 Macro F1，直接盯著它才不會發生「loss 降了但 F1 沒進步」的情況。"}])

    D.h("5.5　DataLoader 在幫你做什麼", 2)
    D.code([
        "train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,",
        "                      num_workers=0)",
    ])
    D.bullet([{"t": "batch_size", "mono": True},
              {"t": "：每次給模型幾張影像。太小會讓 BatchNorm 的統計量不穩，太大會塞爆顯存。"}])
    D.bullet([{"t": "shuffle=True", "mono": True},
              {"t": "：每個 epoch 重新打亂順序。不打亂的話模型會先連續看 210 張 crazing、再看 210 張 inclusion，學到的東西會被順序嚴重影響。驗證集則一律 "},
              {"t": "shuffle=False", "mono": True},
              {"t": "，因為順序不影響指標，固定順序比較好除錯。"}])
    D.bullet([{"t": "num_workers", "mono": True},
              {"t": "：用幾個背景程序去讀檔和做 augmentation。設 0 代表主程序自己做。"}])
    D.note("實際踩到的坑",
           "在 Windows 上 num_workers > 0 時，每個 epoch 都會重新啟動那些背景程序，而每個程序都要重新 import 一次 torch，開銷極大。Stage 2 因此加上 persistent_workers=True 讓它們保持存活，速度差了好幾倍。")

    # ======================================================= 第 6 章
    D.h("第 6 章　Stage 1 逐檔講解", 1, page_break=True)

    D.h("6.1　common.py：所有人的共同地基", 2)
    D.code([
        "CLASS_ORDER = ['crazing', 'inclusion', 'patches',",
        "               'pitted_surface', 'rolled_in_scale', 'scratches']",
        "SEED = 42",
        "NORM_MEAN = [0.5070]",
        "NORM_STD  = [0.2132]",
        "INPUT_SIZE = 128",
    ])
    D.h("為什麼 CLASS_ORDER 要定義在這裡", 3)
    D.para([{"t": "模型輸出的是 0 到 5 的索引，要靠這個 list 翻回類別名稱。訓練時用它把資料夾名稱轉成數字標籤，推論時用它把數字轉回名稱寫進 CSV。"},
            {"t": "如果兩邊順序不一致，每張預測都會對應到錯的類別，Macro F1 直接歸零——而且程式完全不會報錯。", "b": True},
            {"t": "把它放在單一檔案裡，就從根本杜絕了這個風險。"}])
    D.h("為什麼正規化常數要寫死", 3)
    D.para("這兩個數字是用 1,260 張訓練影像算出來的（刻意不含 validation，避免資料洩漏）。寫死而不是每次重算，是因為 TA 跑 benchmark 時只有 hidden 影像、沒有訓練集可以算統計量。如果推論時拿 hidden 資料自己算 mean 和 std，前處理就和訓練時不一致，準確度會明顯下降。")
    D.h("set_seed 為什麼要固定四個隨機源", 3)
    D.code([
        "def set_seed(seed=SEED):",
        "    random.seed(seed)                  # Python 內建，augmentation 用",
        "    np.random.seed(seed)               # NumPy",
        "    torch.manual_seed(seed)            # PyTorch CPU",
        "    torch.cuda.manual_seed_all(seed)   # PyTorch GPU",
        "    torch.backends.cudnn.deterministic = True",
        "    torch.backends.cudnn.benchmark = False",
    ])
    D.para([{"t": "最後兩行特別重要。cuDNN 預設會自動測試好幾種卷積演算法並挑最快的，但不同演算法的浮點累加順序不同，結果會有微小差異。對一般訓練無所謂，但我們要做四組實驗的公平比較，"},
            {"t": "差異必須只來自那兩個開關，不能來自演算法選擇的隨機性。", "b": True}])
    D.h("list_split：為什麼要排序兩次", 3)
    D.code([
        "for idx, cname in enumerate(CLASS_ORDER):   # 第一層：固定類別順序",
        "    for fn in sorted(os.listdir(cdir)):     # 第二層：檔名排序",
    ])
    D.para([{"t": "os.listdir", "mono": True},
            {"t": " 回傳的順序由作業系統決定，不保證穩定。兩層排序確保每次執行拿到的檔案順序完全相同，這樣固定 seed 的 shuffle 才真的能重現。"}])

    D.h("6.2　model.py：Baseline CNN", 2)
    D.para("架構細節第 4 章已經講完，這裡補充兩個規格面的重點。")
    D.bullet([{"t": "必須 from scratch。", "b": True},
              {"t": "規格禁止使用 pretrained weights，也禁止直接套用 ResNet / VGG / MobileNet 的架構。所以這個網路是純粹的 plain CNN——沒有 residual connection、沒有 inverted bottleneck，只有最基本的 Conv、ReLU、MaxPool、Linear。"}])
    D.bullet([{"t": "參數量要可報告。", "b": True},
              {"t": "count_parameters() 回傳總參數和可訓練參數。不開 BN 是 615,270，開 BN 是 615,974——多出來的 704 個就是七層 BN 的 γ 和 β。BN 幾乎不增加模型大小，這是報告裡的一個論點。"}])

    D.h("6.3　dataset.py：兩個開關中的第一個", 2)
    D.code([
        "AUG_CONFIG = {",
        "    'random_horizontal_flip_p': 0.5,",
        "    'random_vertical_flip_p': 0.5,",
        "    'random_rotation_degrees': 15,",
        "    'random_resized_crop_scale': (0.75, 1.0),",
        "    'random_resized_crop_ratio': (0.9, 1.111),",
        "}",
    ])
    D.para("把參數集中成一個 dict，而不是散落在程式碼各處，是為了讓它可以被原封不動存進 checkpoint、寫進 config.json、再印進報告表格。設定只有一個真相來源，就不可能發生「報告寫 15 度但程式跑 10 度」。")
    D.code([
        "def eval_transform(size=INPUT_SIZE):",
        "    return transforms.Compose([",
        "        transforms.Resize((size, size)),",
        "        transforms.ToTensor(),",
        "        transforms.Normalize(NORM_MEAN, NORM_STD)])",
    ], "規格允許的「基本 preprocessing」就是這三件事")
    D.para([{"t": "這條路徑同時被 validation 和 benchmark 推論使用。而 "},
            {"t": "train_transform(augment=False)", "mono": True},
            {"t": " 直接回傳它，所以 E1（不開 augmentation）的訓練和評估走的是完全相同的前處理。"}])
    D.h("ToTensor 其實做了三件事", 3)
    D.para("它把 PIL 影像轉成 PyTorch 張量、把數值從 0–255 的整數縮放到 0–1 的浮點數、並且把維度從 (H, W, C) 轉成 (C, H, W)。很多人以為它只是型別轉換，其實那個縮放很關鍵——Normalize 的 mean 和 std 都是以 0–1 為前提算出來的。")

    D.h("6.4　train.py：四組實驗共用一支程式", 2)
    D.code([
        "p.add_argument('--augment', action='store_true')  # Improvement A",
        "p.add_argument('--bn',      action='store_true')  # Improvement B",
    ])
    D.para("四組 ablation 就是這兩個旗標的四種組合：")
    D.code([
        "python train.py --exp E1_baseline                  # 不開  不開",
        "python train.py --exp E2_aug      --augment        # 開    不開",
        "python train.py --exp E3_bn                  --bn  # 不開  開",
        "python train.py --exp E4_aug_bn   --augment  --bn  # 開    開",
    ])
    D.h("一個容易被忽略但很重要的細節", 3)
    D.code([
        "tr_eval_ld = DataLoader(",
        "    SteelDataset(train_ds.paths, train_ds.labels, val_ds.transform),",
        "    batch_size=args.batch_size, shuffle=False)",
        "trp, trt, _ = predict(model, tr_eval_ld, device)",
        "train_metrics = compute_metrics(trt, trp)",
    ])
    D.para([{"t": "訓練集的準確度是用「evaluation transform」重新算一次的，不是直接拿訓練過程中的 train_acc。"},
            {"t": "因為訓練時的準確度是在加了隨機擾動的影像上算的，會系統性偏低；拿它去和 validation 比，會誤判 overfitting 的程度。", "b": True},
            {"t": "報告裡的 Gap 欄位（train acc 減 val acc）必須建立在同一個前處理基準上才有意義。"}])
    D.h("輸出了什麼", 3)
    D.bullet([{"t": "E4_aug_bn.pth", "mono": True},
              {"t": "：權重加上完整設定（架構開關、augmentation 參數、lr、seed…）。設定跟著權重走，推論時才能正確重建網路。"}])
    D.bullet([{"t": "E4_aug_bn_summary.json", "mono": True},
              {"t": "：訓練曲線、各項指標、confusion matrix、參數量、延遲。"},
              {"t": "報告產生器只讀這份 JSON，所以報告裡每個數字都保證來自實際執行結果。", "b": True}])

    D.h("6.5　evaluate.py：指標計算", 2)
    D.code([
        "f1_score(y_true, y_pred, labels=list(range(6)),",
        "         average='macro', zero_division=0)",
    ])
    D.note("這兩個參數不能省",
           "labels=range(6) 強制把六個類別都納入計算；zero_division=0 對應規格那句「某類無有效預測時該類 F1 = 0」。不加的話，sklearn 遇到完全沒被預測到的類別會直接忽略它，Macro F1 就會被高估——報告數字會比實際好看，但 TA 跑出來不是那樣。")
    D.h("量測延遲的兩個坑", 3)
    D.code([
        "for _ in range(10):              # 坑一：暖身",
        "    model(x)",
        "if device.type == 'cuda':",
        "    torch.cuda.synchronize()     # 坑二：等 GPU 真的算完",
        "t0 = time.time()",
    ])
    D.para("GPU 第一次執行某個運算時要編譯 kernel、配置記憶體，所以要先空跑 10 次。另外 CUDA 是非同步的——Python 發出指令後立刻返回，不等 GPU 算完。不呼叫 synchronize 的話，計時器會在 GPU 還在忙的時候就停掉，量出來的數字會荒謬地小。")

    D.h("6.6　stress_eval.py：為什麼要額外寫一支評估程式", 2)
    D.para("這是 Stage 1 最關鍵的設計決定，背後是一個實際遇到的問題。")
    D.para("四組實驗跑完後，clean validation 的 Macro F1 分別是 1.0000、0.9851、0.9963、1.0000——差距比隨機波動還小，根本無法判斷哪個方法好。追查後發現原因不在模型而在資料：公開 validation 的 270 張裡，有 26 張與某張訓練影像的相似度超過 0.95，甚至有一組是位元組完全相同的檔案。")
    D.figure(F("split_similarity.png"),
             "圖 6-1　validation 影像對訓練集的最大相似度分佈。右側長尾就是近重複影像。", 5.8)
    D.para("split 是課程指定、不可更動的，我們也沒有動它。解法是在同一批 270 張影像上套用四種可重現的劣化，再重新評估那四個 checkpoint：")
    D.code([
        "class CorruptedVal(Dataset):",
        "    def __getitem__(self, i):",
        "        img = Image.open(self.paths[i]).convert('L')",
        "        rng = np.random.default_rng(self.seed + i)   # 關鍵這一行",
        "        img = self.fn(img, rng)",
    ])
    D.note("seed + i 這個寫法為什麼重要",
           "每張影像用「固定基準 + 自己的索引」建立獨立的亂數產生器，所以第 7 張影像受到的劣化永遠一模一樣，不管跑幾次、不管在評估哪個 checkpoint。四個模型面對的是完全相同的 270 張劣化影像，比較才成立。如果用全域 random，評估順序一改結果就變了。")
    D.para("結果成功把四組分開了，而且得到一個反直覺的發現：")
    D.figure(F("stage1/stress_summary.png"),
             "圖 6-2　四組實驗在劣化條件下的平均 Macro F1。", 5.0)
    D.para("A 單獨加入是 −0.0074、B 單獨加入是 −0.0230，兩個都比 baseline 差；但 A 和 B 一起用卻是 +0.0346，交互作用項高達 +0.0649。兩個方法是互補而非可加——augmentation 讓模型看到更多變異，但沒有 BN 時這些變異會讓各層輸入分佈飄移、訓練不穩；BN 把分佈標準化之後，augmentation 的樣本多樣性才真正轉化成泛化能力。")
    D.para("這正好說明了題目為什麼要求四組 ablation，而不是只比較頭尾兩組。")

    D.h("6.7　traditional_baseline.py：不用深度學習的對照組", 2)
    D.code([
        "steps = [('scaler', StandardScaler()),",
        "         ('pca', PCA(n_components=hp['pca'], whiten=True)),",
        "         ('clf', SVC(...))]        # 或 KNeighborsClassifier(...)",
        "pipe = Pipeline(steps)",
    ])
    D.para([{"t": "兩種方法共用完全相同的前處理，只替換最後一個步驟，所以效能差異純粹來自分類器本身。用 sklearn 的 "},
            {"t": "Pipeline", "mono": True},
            {"t": " 包起來還有一個隱性好處：PCA 和 scaler 只會在訓練資料上 fit，預測時自動只做 transform，不會把驗證集的統計量洩漏進去。"}])
    D.para("結果是 PCA+SVM 得到 0.7843，PCA+KNN 只有 0.4537。差這麼多的原因是 PCA 給的是全域灰階強度的線性主成分，KNN 直接在這個空間比歐式距離，對整體亮度變化極度敏感；SVM 的 RBF kernel 至少能學非線性邊界。但兩者都沒有任何「局部紋理」的概念——而 crazing 和 rolled_in_scale 的差別正是局部紋理的方向。這就是 CNN 能跳到 1.0000 的根本原因。")

    D.h("6.8　stage1_inference.py：TA 唯一會執行的檔案", 2)
    D.code([
        "HERE = os.path.dirname(os.path.abspath(__file__))",
        "sys.path.insert(0, os.path.join(HERE, 'src'))",
        "CHECKPOINT = os.path.join(HERE, 'stage1_model.pth')",
    ])
    D.note("為什麼要用 __file__ 而不是相對路徑",
           "規格明文要求「inference script 必須自行尋找同一 submission 根目錄內的固定 checkpoint；不可依目前 working directory 假設路徑」。如果寫成 'stage1_model.pth'，TA 從別的目錄執行就會找不到檔案，benchmark 直接失敗。__file__ 是這個 .py 檔自己的位置，不管從哪裡執行都正確。")
    D.code([
        "ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)",
        "cfg = ckpt.get('config', {})",
        "model = build_model(use_bn=cfg.get('bn', True),",
        "                    dropout=cfg.get('dropout', 0.0))",
        "model.load_state_dict(ckpt['state_dict'])",
        "model.to(device).eval()",
    ])
    D.para([{"t": "架構參數是從 checkpoint 自己帶的 config 讀出來的，而不是寫死在推論腳本裡。這樣就算之後換設定重新訓練，推論腳本也不必跟著改——"},
            {"t": "架構和權重永遠對得上。", "b": True},
            {"t": " 另外 "}, {"t": "map_location='cpu'", "mono": True},
            {"t": " 讓它在沒有 GPU 的機器上也能載入。"}])
    D.code([
        "rows.extend((os.path.splitext(f)[0], CLASS_ORDER[p])",
        "            for f, p in zip(chunk, pred))",
    ])
    D.para([{"t": "os.path.splitext(f)[0]", "mono": True},
            {"t": " 取的是「檔名去掉副檔名」，也就是規格說的 filename stem：HID_0001.png 變成 HID_0001。寫錯的話每一列都對不上，Macro F1 直接 0。"}])

    # ======================================================= 第 7 章
    D.h("第 7 章　Stage 2：遷移學習的程式碼", 1, page_break=True)
    D.para("Stage 2 的檔案分層和 Stage 1 完全一樣，所以這一章只講不同的地方。")

    D.h("7.1　怎麼載入預訓練模型並換掉分類頭", 2)
    D.code([
        "def build_backbone(arch, pretrained=True):",
        "    if arch == 'resnet18':",
        "        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None",
        "        m = models.resnet18(weights=w)",
        "        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)   # 換頭",
        "        return m, 'fc'",
    ])
    D.para([{"t": "ImageNet 的分類頭輸出 1000 類，我們只要 6 類，所以把最後那層 Linear 整個換掉。"},
            {"t": "m.fc.in_features", "mono": True},
            {"t": " 會自動讀出原本那層的輸入維度（ResNet18 是 512），這樣就不用手動查數字，換成別的架構也能通用。"}])
    D.note("為什麼回傳 'fc' 這個字串",
           "不同架構的分類頭屬性名稱不一樣：ResNet 叫 fc，DenseNet 和 MobileNet 叫 classifier。把名稱一起回傳，後面凍結參數和 L2-SP 才知道哪些參數屬於新的頭、哪些屬於 backbone。")
    D.para([{"t": "推論時則一定要用 "}, {"t": "pretrained=False", "mono": True},
            {"t": "：權重全部來自我們自己的 checkpoint，執行期絕對不能連網下載——規格明文禁止。"}])

    D.h("7.2　「凍結參數」在程式碼裡只是一行", 2)
    D.figure(F("doc/freeze.png"), "圖 7-1　兩種 fine-tuning 策略的差別。", 6.0)
    D.code([
        "if strategy == 'feature_extraction':",
        "    for n, p in model.named_parameters():",
        "        p.requires_grad = n.startswith(head_attr)",
    ])
    D.para([{"t": "requires_grad = False", "mono": True},
            {"t": " 的意思是「這個參數不需要梯度」。反向傳播時 PyTorch 會跳過它，optimizer 也就不會更新它。這一行的效果是：ResNet18 的 1,117 萬個參數裡，只有分類頭的 3,078 個會被訓練，佔 0.03%。"}])
    D.code([
        "params = [p for p in model.parameters() if p.requires_grad]",
        "optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=wd)",
    ])
    D.note("這個過濾也不能省",
           "如果把全部參數都丟給 optimizer（包含凍結的），AdamW 這類帶 weight decay 的優化器仍然會對它們做 weight decay，等於偷偷改動了你以為凍結的權重。只傳 requires_grad=True 的參數才是真的凍結。")

    D.h("7.3　L2-SP：最終選用的策略", 2)
    D.para("L2-SP 的想法很簡單：全部的層都訓練，但在 loss 裡加一項懲罰，讓 backbone 的權重不要離 ImageNet 的解太遠。")
    D.code([
        "class L2SP:",
        "    def __init__(self, model, head_attr, alpha=0.01, beta=0.01):",
        "        self.ref = {n: p.detach().clone()      # 先記住預訓練權重",
        "                    for n, p in model.named_parameters()",
        "                    if not n.startswith(head_attr)}",
        "",
        "    def __call__(self, model):",
        "        loss = 0.0",
        "        for n, p in model.named_parameters():",
        "            if n in self.ref:",
        "                loss += self.alpha * 0.5 * ((p - self.ref[n])**2).sum()",
        "            else:",
        "                loss += self.beta * 0.5 * (p ** 2).sum()",
        "        return loss",
    ])
    D.para("數學式是 α/2 · ‖w − w₀‖² + β/2 · ‖w_head‖²。backbone 被拉向預訓練解 w₀，新的分類頭則用一般的 L2（因為它是隨機初始化的，沒有「原本的解」可以拉）。")
    D.para([{"t": "那個 "}, {"t": ".detach().clone()", "mono": True},
            {"t": " 很重要：detach 切斷計算圖（這份參考權重不需要梯度），clone 複製一份記憶體（否則它會跟著模型一起被更新，懲罰項就永遠是 0）。"}])
    D.para("在只有 1,260 張訓練影像的情況下，這個約束有效防止 backbone 被少量資料帶偏。實驗結果它在三種劣化下都最穩定（Robust Macro F1 0.9790，全場最高），因此成為 Stage 2 的最終模型。")

    D.h("7.4　把 robustness 設計進 augmentation", 2)
    D.para("Stage 2 有一個特殊要求：TA 會用同一份 checkpoint 跑四次，分別是正常影像和三種劣化版本，而且禁止依資料夾名稱切換行為。既然不能在推論時做任何事，唯一的辦法就是在訓練時讓模型見過這些劣化。")
    D.table(["對應的 Benchmark", "Augmentation", "模擬什麼"], [
        ["lighting", "ColorJitter（亮度 / 對比）＋ RandomGamma", "照明老化、角度改變、反光"],
        ["optical", "GaussianBlur ＋ 降取樣再放大", "鏡頭失焦、鏡頭髒污"],
        ["sensor_compression", "JPEG 重新壓縮 ＋ 高斯雜訊", "影像傳輸壓縮、感測器雜訊"],
        ["一般泛化", "RandomResizedCrop / Flip / Rotation / RandomErasing", "取景與角度變化"],
    ], [140, 230, 200])
    D.code([
        "class RandomJPEG:",
        "    def __call__(self, img):",
        "        if random.random() < self.p:",
        "            buf = io.BytesIO()",
        "            img.save(buf, format='JPEG',",
        "                     quality=random.randint(*self.quality))",
        "            buf.seek(0)",
        "            img = Image.open(buf)",
        "        return img",
    ], "自訂 transform：用記憶體緩衝區模擬 JPEG 壓縮失真")
    D.para([{"t": "torchvision 沒有內建的 JPEG augmentation，所以自己寫一個。"},
            {"t": "io.BytesIO()", "mono": True},
            {"t": " 是「假裝成檔案的記憶體區塊」——把影像以 JPEG 格式存進去再讀出來，就得到真實的壓縮失真，而且完全不碰硬碟。"}])
    D.note("寫自訂 transform 的規則",
           "只要是一個「可呼叫的物件」就行——定義 __call__ 方法，接收一張影像、回傳一張影像，就能直接放進 transforms.Compose 的清單裡。這是 torchvision 的設計，不需要繼承任何基底類別。")
    D.para("這套 augmentation 在四組實驗中完全相同，所以不影響公平比較；推論端則一律只做 Resize / ToTensor / Normalize，對所有輸入資料夾一視同仁。")
    D.figure(F("stage2/robustness.png"),
             "圖 7-2　各模型在乾淨與三種劣化條件下的表現。", 6.0)

    # ======================================================= 第 8 章
    D.h("第 8 章　Stage 3：物件偵測與自己實作 mAP", 1, page_break=True)

    D.h("8.1　為什麼 model.py 只有 49 行", 2)
    D.code([
        "def load_detector(weights):",
        "    from ultralytics import YOLO",
        "    return YOLO(weights)",
    ])
    D.para([{"t": "Stage 3 用的是 Ultralytics 的 YOLOv8s，架構由套件提供，不需要自己刻。但還是要寫這個封裝，理由是："},
            {"t": "訓練、評估、benchmark 三個地方都透過同一個函式載入模型。", "b": True},
            {"t": " 如果各自寫 YOLO(...)，日後要加參數或換版本就得改三個地方，很容易漏掉一個。"}])

    D.h("8.2　prepare_data.py：資料格式的轉接頭", 2)
    D.para([{"t": "公開資料集已經附了 YOLO 格式的標註，但放在 "},
            {"t": "train/labels_yolo/", "mono": True},
            {"t": "；而 Ultralytics 規定要用「把路徑裡的 /images/ 換成 /labels/」來找標註檔。路徑對不上，它就會認為所有影像都沒有標註，訓練出一個什麼都偵測不到的模型。"}])
    D.para("所以要有這支程式，把公開 split 複製成 Ultralytics 要的結構。注意它只是複製、不重新切分——TA 指定的 split 必須原封不動。")
    D.code([
        "lbl = os.path.join(src_lbl, stem + '.txt')",
        "if os.path.exists(lbl):",
        "    shutil.copyfile(lbl, dst)",
        "else:",
        "    open(dst, 'w').close()      # 沒有物件的影像要給一個空檔",
    ])
    D.note("空標註檔不能省",
           "如果某張影像沒有任何標註，Ultralytics 需要看到一個「存在但內容是空的」.txt，才知道這張圖是背景影像。完全沒有這個檔的話它會當成資料損壞而跳過，訓練集就少了資料。")

    D.h("8.3　自己實作 mAP@0.5", 2)
    D.para("這是 Stage 3 最花心思的一段。題目給了非常精確的評分定義，而各家框架的 mAP 實作細節都不一樣，所以我們照定義自己寫一份。")
    D.figure(F("doc/map_steps.png"), "圖 8-1　題目規定的七個步驟。", 6.0)
    D.h("步驟 ③④：貪婪配對", 3)
    D.code([
        "for i, (iid, _score, box) in enumerate(rows):  # 已依 confidence 排序",
        "    ious = iou_matrix(np.array([box]), rec['boxes'])[0]",
        "    j = int(np.argmax(ious))",
        "    if j >= 0 and ious[j] >= iou_thr and not rec['used'][j]:",
        "        rec['used'][j] = True    # 標記這個 GT 已經被認領",
        "        tp[i] = 1",
        "    else:",
        "        fp[i] = 1",
    ])
    D.para([{"t": "rec['used']", "mono": True},
            {"t": " 這個布林陣列是整段程式的核心。它實作了題目那句「一個 Ground Truth 最多只能匹配一個 prediction；同一 GT 的後續重複框皆計為 False Positive」。因為預測已經依 confidence 由高到低排序，先被配對到的一定是分數最高的那個，這就是「貪婪」的意思。"}])
    D.h("步驟 ⑥：all-points 插值", 3)
    D.code([
        "mrec = np.concatenate(([0.0], recall, [1.0]))",
        "mpre = np.concatenate(([0.0], precision, [0.0]))",
        "for i in range(len(mpre) - 2, -1, -1):        # 由右往左",
        "    mpre[i] = max(mpre[i], mpre[i + 1])       # 做成單調遞減的階梯",
        "idx = np.where(mrec[1:] != mrec[:-1])[0]      # recall 有變化的位置",
        "return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])",
    ])
    D.para("Precision-Recall 曲線是鋸齒狀的。VOC 2010+ 的規定是：先把 precision 由右往左取累積最大值，把曲線變成單調遞減的階梯，再計算階梯下的面積。那個由後往前的迴圈就是在做這件事。")
    D.h("為什麼要自己寫而不用現成的", 3)
    D.table(["mAP@0.5 的來源", "數值", "差異"], [
        ["本專案（依題目定義：VOC 2010+ all-points）", "0.7419", "—"],
        ["Ultralytics 內建 validator（101-point 插值）", "0.7588", "+0.0169"],
    ], [300, 110, 110])
    D.para([{"t": "差了 0.0169，而且套件的數字比較高。正式成績一律以 TA runner 為準，所以報告全部引用自己實作的版本——"},
            {"t": "引用套件內建值會系統性高估自己的成績。", "b": True}])
    D.para("為了確保這份實作真的正確，另外寫了 9 個單元測試：")
    D.code([
        "check('duplicate box on the same GT counts as FP', ..., 0.8333)",
        "check('wrong class -> FP for that class', ..., 0.0)",
        "check('IoU 0.493 < 0.5', ..., 0.0)",
        "check('IoU 0.515 > 0.5', ..., 1.0)",
        "check('VOC2010+ all-points interpolation', ..., 0.76)",
    ], "test_map.py：每個測試對應定義裡的一條規則")
    D.para("每一個測試都是手算出正確答案、再和程式輸出比對。這樣才敢說「我們的評分程式和 TA 的定義一致」。")

    D.h("8.4　stage3_detect.py 的三個門檻", 2)
    D.code([
        "CONF_THRESHOLD = 0.001    # 很低，刻意的",
        "NMS_IOU_THRESHOLD = 0.6",
        "MAX_DET = 100             # 格式硬性上限",
    ])
    D.para("confidence 設這麼低是反直覺的，但有實測根據：")
    D.table(["Confidence", "max_det", "NMS IoU", "mAP@0.5", "平均框數/張"], [
        ["0.001", "100", "0.60", "0.7419", "39.4"],
        ["0.005", "100", "0.60", "0.7403", "16.3"],
        ["0.010", "100", "0.60", "0.7383", "11.6"],
        ["0.050", "100", "0.60", "0.7170", "5.3"],
        ["0.001", "100", "0.70", "0.7319", "55.8"],
    ], [100, 85, 85, 100, 110])
    D.para("道理是：AP 是對整條 PR 曲線積分，低分的框只會延伸曲線的右端（提高 recall 上限），不會傷害已經算好的高分段。先把低分框濾掉，等於自己把曲線截斷。至於 NMS 門檻調高到 0.7 反而變差，是因為同一個瑕疵上的重複框依規則一律算 False Positive。")
    D.note("這不代表產線上也該用 0.001",
           "0.001 是「給評分器看」的設定。真正要給作業員看的門檻應該在下游 UI 設定（實測 F1 最佳落在 0.30 附近）。兩者目的不同，不該混為一談。")
    D.figure(F("stage3/gt_vs_pred.png"),
             "圖 8-2　Ground Truth（綠）與 Prediction（紅）對照。", 6.3)

    # ======================================================= 第 9 章
    D.h("第 9 章　報告產生器與繳交驗證", 1, page_break=True)

    D.h("9.1　為什麼報告是程式產生的", 2)
    D.para("四份 PDF 都是用 Python 產生的，不是手寫。理由是資料流：")
    D.code([
        "train.py  ->  work/stage1_runs/E4_aug_bn_summary.json",
        "                      |",
        "                      v",
        "          make_stage1_report.py  ->  stage1_report.pdf",
    ])
    D.para([{"t": "報告裡每一個數字都直接讀自那些 JSON，而 JSON 又是訓練程式寫出來的。這樣"},
            {"t": "不可能出現「報告寫 0.98 但實際是 0.97」的情況", "b": True},
            {"t": "；重新訓練之後重跑產生器，所有數字、表格、圖表自動同步。"}])

    D.h("9.2　中文字型的一個實際問題", 2)
    D.para([{"t": "微軟正黑體沒有收錄打勾、打叉、減號、大於等於這幾個符號。直接用的話 PDF 會印出空白方框。所以 "},
            {"t": "report_builder.py", "mono": True},
            {"t": " 加了一層自動替換，並在產生 PDF 時掃描所有文字，遇到字型沒有的字就在終端機印出警告。這樣就不會等到 PDF 印出來才發現缺字。"}])

    D.h("9.3　兩層繳交驗證", 2)
    D.table(["驗證工具", "檢查什麼", "不檢查什麼"], [
        ["validate_submission.py@@（課程官方）",
         "檔名是否齊全、config.json 是否為合法 JSON、src/ 三個檔的語法、CLI 參數是否存在",
         "不會實際執行你的腳本，也不檢查輸出格式"],
        ["check_submission.py@@（自己寫的）",
         "從另一個工作目錄實際執行固定指令、驗證輸出 schema、bbox 範圍、confidence 區間、image_id 對應、每張框數上限",
         "—"],
    ], [150, 300, 150])
    D.para("官方那支 PASS 不代表 benchmark 跑得起來——它完全不會執行你的程式。所以另外寫了一支會真的解壓到暫存目錄、從那裡執行、再逐項驗證輸出的檢查器。兩支都過，才算有把握。")

    # ======================================================= 附錄
    D.h("附錄　PyTorch 慣用法速查", 1, page_break=True)
    D.table(["寫法", "意思", "常見誤解"], [
        ["model(x)", "前向傳播", "不要寫 model.forward(x)，會跳過 nn.Module 的內部機制"],
        ["model.train() / model.eval()", "切換 BatchNorm 和 Dropout 的行為", "不會改變任何權重，只是切換模式"],
        ["torch.no_grad()", "不建立計算圖", "省記憶體也變快，推論時必用"],
        ["optimizer.zero_grad()", "清空梯度", "PyTorch 梯度是累加的，不清會爆炸"],
        ["loss.backward()", "反向傳播算梯度", "只算梯度，不更新參數"],
        ["optimizer.step()", "依梯度更新參數", "沒呼叫它的話參數永遠不變"],
        ["p.requires_grad = False", "凍結參數", "還要在建 optimizer 時過濾，否則 weight decay 仍會改動它"],
        ["tensor.detach()", "切斷計算圖", "常和 .clone() 一起用來保存一份不會被更新的複本"],
        ["x.to(device)", "把資料搬到 GPU", "模型和資料必須在同一個裝置上"],
        ["logit.argmax(1)", "取最高分的類別索引", "參數 1 是類別維度，0 是 batch 維度"],
        ["nn.Sequential(*layers)", "把 list 攤開傳進去", "星號是 Python 的解包語法"],
        ["copy.deepcopy(state_dict())", "真正複製一份權重", "不 deepcopy 會拿到參照，之後會被改掉"],
    ], [180, 145, 290])
    D.para("以上。如果有哪一段想再深入，或想看某個檔案的完整程式碼逐行講解，都可以再問。",
           color=GREY, size=9.5)

    # python-docx ships a settings.xml whose <w:zoom> has no w:percent, which
    # fails OOXML schema validation; give it the required attribute.
    from docx.oxml.ns import qn as _qn
    zoom = d.settings.element.find(_qn("w:zoom"))
    if zoom is not None and zoom.get(_qn("w:percent")) is None:
        zoom.set(_qn("w:percent"), "100")

    d.save(OUT)
    print("wrote", OUT, "%.2f MB" % (os.path.getsize(OUT) / 1024 ** 2))


if __name__ == "__main__":
    main()
