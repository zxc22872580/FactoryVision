# -*- coding: utf-8 -*-
"""Actually execute the notebook's key code cells locally, so we know the
Colab version really runs and is not just syntactically valid.

Install cells and the Drive-mount cell are skipped; DATA_ROOT / WORK are
injected instead. Epoch counts are shrunk so the smoke test stays short.
"""
import io
import json
import os
import re
import sys
import traceback

NB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "FactoryVision_Colab.ipynb")

DATA_ROOT = os.environ.get(
    "FACTORYVISION_DATA",
    r"C:\Users\zxc22\OneDrive\桌面\中興大學\電腦視覺\期末"
    r"\FactoryVision_STUDENT_PUBLIC\STUDENT_PUBLIC")
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_smoke")

# cells to execute, in order (indices from the notebook)
PLAN = [5, 12, 13, 20, 21, 23, 26, 28, 30, 31, 35, 38, 40, 41, 42, 43,
        45, 49]

# make the smoke run fast
SHRINK = [
    (r"^EPOCHS\s*=\s*\d+", "EPOCHS       = 2"),
    (r"^S2_EPOCHS\s*=\s*\d+", "S2_EPOCHS   = 1"),
    (r"^NUM_WORKERS\s*=\s*\d+", "NUM_WORKERS  = 0"),
    (r"^S2_BATCH\s*=\s*\d+", "S2_BATCH    = 16"),
    (r"^PCA_DIM\s*=\s*\d+", "PCA_DIM   = 32"),
]
SHELL = re.compile(r"^\s*[!]")


def prep(src):
    out = []
    for ln in src.split("\n"):
        if SHELL.match(ln):
            continue
        for pat, repl in SHRINK:
            if re.match(pat, ln):
                ln = repl + "   # (smoke test override)"
                break
        out.append(ln)
    return "\n".join(out)


def main():
    nb = json.load(io.open(NB, encoding="utf-8"))
    cells = nb["cells"]

    import matplotlib
    matplotlib.use("Agg")           # 不開視窗
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: plt.close("all")

    g = {"__name__": "__main__"}
    # 注入 Colab 那邊由環境提供的變數
    os.makedirs(WORK, exist_ok=True)
    exec("import os, glob\n"
         "DATA_ROOT = %r\nWORK = %r\n"
         "cls_dir = os.path.join(DATA_ROOT, 'classification')\n"
         "det_dir = os.path.join(DATA_ROOT, 'detection')\n"
         "CLASS_ORDER = ['crazing','inclusion','patches','pitted_surface',"
         "'rolled_in_scale','scratches']\n"
         % (DATA_ROOT, WORK), g)
    # 字型（本機用微軟正黑體，Colab 用 Noto）
    exec("from matplotlib import font_manager, rcParams\n"
         "p = r'C:\\\\Windows\\\\Fonts\\\\msjh.ttc'\n"
         "if os.path.exists(p):\n"
         "    font_manager.fontManager.addfont(p)\n"
         "    rcParams['font.sans-serif'] = ['Microsoft JhengHei']\n"
         "rcParams['axes.unicode_minus'] = False\n", g)

    ok = fail = 0
    for idx in PLAN:
        c = cells[idx]
        first = [l.strip() for l in c["source"] if l.strip()][0][:60]
        src = prep("".join(c["source"]))
        print("\n===== cell %d  %s" % (idx, first))
        try:
            exec(compile(src, "<cell %d>" % idx, "exec"), g)
            ok += 1
            print("  [OK]")
        except Exception:
            fail += 1
            print("  [FAIL]")
            traceback.print_exc(limit=4)
            break
    print("\n==== 執行 %d 格成功, %d 格失敗 ====" % (ok, fail))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
