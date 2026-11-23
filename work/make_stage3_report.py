# -*- coding: utf-8 -*-
"""Generate Team13_Stage3/stage3_report.pdf from the detection artifacts."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plots as PL
import report_builder as RB
from report_builder import (H, P, PageBreak, Spacer, bullets, figure_block,
                            fmt, mm, table)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures", "stage3")
OUT = os.path.join(ROOT, "Team13_Stage3", "stage3_report.pdf")
TEAM = "Team13"
CLS = PL.CLASS_ORDER


def main():
    R = json.load(open(os.path.join(HERE, "outputs",
                                    "stage3_val_results.json"),
                       encoding="utf-8"))
    DS = json.load(open(os.path.join(HERE, "outputs",
                                     "stage3_dataset_stats.json"),
                        encoding="utf-8"))
    tcfg_path = os.path.join(HERE, "outputs", "stage3_train_config.json")
    T = json.load(open(tcfg_path, encoding="utf-8")) \
        if os.path.exists(tcfg_path) else {}
    val = R["validation"]

    st = []
    st += [P("FactoryVision Challenge — Stage 3 報告", "title"),
           P("Object Detection｜鋼材表面瑕疵定位<br/>{}　random seed = 42"
             .format(TEAM), "subtitle"), Spacer(1, 6 * mm)]

    st += [H("1. 任務與資料", 1)]
    st += [P("Stage 3 把任務從「這是什麼瑕疵」延伸到「瑕疵在哪裡」。"
             "使用 TA 提供的固定 split：Train 1,260 張、Validation 270 張，"
             "Hidden Test 270 張僅由 TA 保管。標註為 VOC bounding box 與"
             "六個 class。")]
    dall = DS["all"]
    st += [table([["統計項目", "數值"],
                  ["影像數（train + val）", str(dall["images"])],
                  ["平均每張物件數", fmt(dall["objects_per_image_mean"], 2)],
                  ["單張最多物件數", str(dall["objects_per_image_max"])],
                  ["Bounding box 總數",
                   str(sum(v["n_boxes"] for v in dall["per_class"].values()))]],
                 col_widths=[60 * mm, 114 * mm])]
    rows = [["Class", "Box 數", "相對面積中位數", "面積 < 3% 的比例",
             "長寬比中位數"]]
    for c in CLS:
        v = dall["per_class"][c]
        rows.append([c, str(v["n_boxes"]), fmt(v["rel_area_median"], 3),
                     "{:.1f}%".format(100 * v["small_box_ratio"]),
                     fmt(v["aspect_median"], 2)])
    st += [Spacer(1, 2 * mm), table(rows, col_widths=[36 * mm, 24 * mm,
                                                      36 * mm, 38 * mm,
                                                      34 * mm])]
    st += [Spacer(1, 2 * mm), figure_block(
        os.path.join(HERE, "figures", "stage3_dataset_stats.png"), 172,
        "圖 1：資料集統計。")]
    st += [P("這張表直接預示了後面的失敗型態：inclusion 有 {:.0f}% 的框小於影像面積的 3%，"
             "是最難的小物件；scratches 的長寬比中位數只有 {:.2f}（極細長的垂直刮痕）；"
             "pitted_surface 的框卻佔了影像面積的 {:.0f}%，幾乎是整張圖。"
             "同一個 detector 必須同時處理這兩個極端。"
             .format(100 * dall["per_class"]["inclusion"]["small_box_ratio"],
                     dall["per_class"]["scratches"]["aspect_median"],
                     100 * dall["per_class"]["pitted_surface"]["rel_area_median"]))]
    st += [P("註：NEU-DET 的 VOC 標註把該類別寫成 rolled-in_scale（連字號），"
             "而 benchmark 規定的輸出名稱是 rolled_in_scale（底線）。"
             "本專案在讀取標註時統一正規化，輸出一律使用底線版本。", "note")]

    st += [PageBreak(), H("2. Detector 與訓練設定", 1)]
    aug = T.get("augmentation", {})
    st += [table([
        ["項目", "設定"],
        ["Detector", T.get("detector", "YOLOv8s")],
        ["Backbone", T.get("backbone", "CSPDarknet (YOLOv8s)")],
        ["Pretrained source", T.get("pretrained_source",
                                    "Ultralytics YOLOv8s COCO 權重")],
        ["參數量", "{:,}".format(R["model"]["params"])],
        ["Input size", "{}×{}".format(R["imgsz"], R["imgsz"])],
        ["Epochs / Batch", "{} / {}".format(T.get("epochs"), T.get("batch"))],
        ["Optimizer / lr0 / lrf", "{} / {} / {}".format(
            T.get("optimizer"), T.get("lr0"), T.get("lrf"))],
        ["Early stopping patience", str(T.get("patience"))],
        ["Augmentation", "rotation ±{}°, translate {}, scale {}, fliplr {}, "
                         "flipud {}, mosaic {}（最後 {} epoch 關閉）, HSV-V {}"
            .format(aug.get("degrees"), aug.get("translate"),
                    aug.get("scale"), aug.get("fliplr"), aug.get("flipud"),
                    aug.get("mosaic"), aug.get("close_mosaic"),
                    aug.get("hsv_v"))],
        ["Seed", str(T.get("seed", 42))],
        ["Checkpoint 大小", "{} MB".format(fmt(R["checkpoint_mb"], 1))],
    ], col_widths=[44 * mm, 130 * mm])]
    st += [P("選擇 one-stage anchor-free detector 的理由：影像只有 200×200、"
             "每張平均不到 2.4 個物件，two-stage 的 region proposal 帶來的收益"
             "有限卻顯著增加延遲；anchor-free head 也免去為「極細長的 scratches」"
             "與「幾乎滿版的 crazing / pitted_surface」手動調 anchor 尺寸的問題。"
             "COCO pretrained 權重只是通用物件偵測器，並非在 NEU-DET 上 fine-tune "
             "過的 checkpoint。")]

    st += [H("2.1 IoU / Confidence / NMS 的作用", 2)]
    st += bullets([
        "<b>IoU</b>：預測框與 ground-truth 框的交集除以聯集。評分時 IoU ≥ 0.5 "
        "且類別正確才算 True Positive，否則一律算 False Positive。",
        "<b>Confidence Threshold</b>：低於門檻的框直接不輸出。提交時固定為 "
        "{}，刻意取很低。".format(R["conf_threshold"]),
        "<b>NMS（Non-Maximum Suppression）</b>：同一類別中，若兩個框的 IoU 超過 "
        "{}，只保留分數較高者，用來消除同一瑕疵上的重複框——這很重要，因為"
        "評分規則規定「一個 GT 最多只能匹配一個 prediction，重複框一律算 FP」。"
        .format(R["nms_iou_threshold"]),
        "<b>每張最多 {} 個 detections</b>：超過即視為格式錯誤，程式按 confidence "
        "由高到低截斷。".format(R["max_det"]),
    ])

    st += [PageBreak(), H("3. Validation 評估結果", 1)]
    st += [P("以下所有數字都由自行實作、完全依照題目定義的評分程式算出："
             "IoU 門檻 0.5、每類依 confidence 排序、一個 GT 只能被匹配一次、"
             "重複框算 FP、每類 AP 使用 Pascal VOC 2010+ all-points interpolation、"
             "六類 AP 不加權平均。")]
    rows = [["Class", "AP@0.5", "GT 數", "預測數",
             "Precision @{:.2f}".format(R["report_conf"]),
             "Recall @{:.2f}".format(R["report_conf"]), "TP", "FP", "FN"]]
    for c in CLS:
        v = val["per_class"][c]
        rows.append([c, fmt(v["ap50"]), str(v["n_gt"]), str(v["n_pred"]),
                     fmt(v["precision_at_thr"], 3),
                     fmt(v["recall_at_thr"], 3), "{:.0f}".format(v["tp_at_thr"]),
                     "{:.0f}".format(v["fp_at_thr"]),
                     "{:.0f}".format(v["fn_at_thr"])])
    rows.append(["<b>mAP@0.5 / micro</b>", "<b>{}</b>".format(fmt(val["map50"])),
                 str(val["n_gt_total"]), "—",
                 "<b>{}</b>".format(fmt(val["micro_precision_at_thr"], 3)),
                 "<b>{}</b>".format(fmt(val["micro_recall_at_thr"], 3)),
                 "—", "—", "—"])
    st += [table(rows, col_widths=[32 * mm, 19 * mm, 15 * mm, 16 * mm,
                                   25 * mm, 23 * mm, 14 * mm, 14 * mm,
                                   14 * mm], highlight_rows=(len(rows) - 1,),
                 font_size=7.6)]
    st += [Spacer(1, 2 * mm), table([
        ["效能項目", "數值"],
        ["單張推論（含前處理 + forward + NMS，batch = 1）",
         "{} ms".format(fmt(R["latency_ms_single"], 1))],
        ["批次推論平均（batch = 8）",
         "{} ms / 張".format(fmt(R["latency_ms_batched"], 1))],
        ["等效吞吐量", "{} FPS（batch = 1）".format(
            fmt(1000.0 / R["latency_ms_single"], 1))],
        ["Checkpoint 大小", "{} MB".format(fmt(R["checkpoint_mb"], 1))],
    ], col_widths=[86 * mm, 88 * mm])]
    st += [Spacer(1, 2 * mm),
           figure_block(os.path.join(FIG, "per_class_ap.png"), 135,
                        "圖 2：每一類別的 AP@0.5。")]
    st += [figure_block(os.path.join(FIG, "pr_curves.png"), 130,
                        "圖 3：各類別的 Precision-Recall 曲線（IoU = 0.5）。")]

    st += [PageBreak(), H("4. 輸出參數與 Confidence Threshold 的取捨", 1)]
    sweep = R["threshold_sweep"]
    rows = [["Threshold", "Precision", "Recall", "F1"]]
    for s in sweep:
        rows.append([fmt(s["threshold"], 2), fmt(s["micro_precision"], 3),
                     fmt(s["micro_recall"], 3), fmt(s["f1"], 3)])
    best = max(sweep, key=lambda s: s["f1"])
    hl = 1 + sweep.index(best)
    st += [table(rows, col_widths=[30 * mm, 36 * mm, 36 * mm, 36 * mm],
                 highlight_rows=(hl,))]
    st += [Spacer(1, 2 * mm),
           figure_block(os.path.join(FIG, "threshold_sweep.png"), 135,
                        "圖 4：Confidence Threshold 對 Precision / Recall 的影響。")]
    st += [P("門檻拉高，Precision 上升、Recall 下降，F1 在 {:.2f} 附近達到最佳"
             "（{:.3f}）。但這只影響「人看的輸出」，不影響 mAP——"
             "因為 AP 是對整條 PR 曲線積分，低分框只會延伸曲線右端。"
             .format(best["threshold"], best["f1"]))]
    sw = R.get("setting_sweep") or []
    if sw:
        st += [H("4.1 三個輸出參數的實測影響", 2)]
        st += [P("conf / NMS IoU / max_det 都會直接改變送進評分器的框，"
                 "因此我們在 validation 上實測了各種組合，再把最好的那組"
                 "寫死在 stage3_detect.py 裡。")]
        rows = [["Confidence", "max_det", "NMS IoU", "mAP@0.5",
                 "平均框數/張", "超過 100 框的影像數"]]
        best_i = max(range(len(sw)), key=lambda i: sw[i]["map50"])
        for e in sw:
            rows.append([fmt(e["conf"], 3), str(e["max_det"]),
                         fmt(e["nms_iou"], 2), fmt(e["map50"]),
                         fmt(e["avg_boxes_per_image"], 1),
                         str(e["images_over_100_boxes"])])
        st += [table(rows, col_widths=[28 * mm, 24 * mm, 24 * mm, 26 * mm,
                                       32 * mm, 40 * mm],
                     highlight_rows=(best_i + 1,))]
        st += [P("結論：confidence 越低 mAP 越高（低分框只會延伸 PR 曲線右端，"
                 "不會傷害已經算好的高分段），但平均框數也跟著上升；"
                 "NMS IoU 調高到 0.7 反而變差，因為同一個瑕疵上的重複框"
                 "依規則一律算 False Positive。"
                 "我們最終採用 conf = {}、NMS IoU = {}、max_det = {}。"
                 "程式會依 confidence 由高到低截斷到 {} 個框，"
                 "因此即使某張圖的框數超過上限也不會違反格式。"
                 .format(R["conf_threshold"], R["nms_iou_threshold"],
                         R["max_det"], R["max_det"]))]

    ur = R.get("ultralytics_reference")
    if ur:
        st += [H("4.2 為什麼我們自己實作評分程式", 2)]
        st += [table([
            ["mAP@0.5 的來源", "數值", "差異"],
            ["本報告（依題目定義：VOC 2010+ all-points）", fmt(R["validation"]["map50"]), "—"],
            ["Ultralytics 內建 validator（101-point 插值）", fmt(ur["map50"]),
             "{:+.4f}".format(ur["map50"] - R["validation"]["map50"])],
        ], col_widths=[82 * mm, 46 * mm, 46 * mm], highlight_rows=(1,))]
        st += [P("兩者相差 {:.4f}，而且 Ultralytics 的數字比較高。"
                 "差異來自 AP 的積分方式（101-point 插值會把稀疏的 PR 曲線"
                 "墊高）與匹配細節。由於正式成績一律以 TA runner 為準，"
                 "<b>本報告所有 mAP 都引用我們自己依題目定義實作的版本</b>，"
                 "不引用 framework 內建值——否則會系統性高估自己的成績。"
                 "我們也為這支評分程式寫了 9 個單元測試"
                 "（src/test_map.py），逐條驗證重複框算 FP、類別錯算 FP、"
                 "IoU 門檻精確落在 0.5、以及 all-points 積分的正確性。"
                 .format(abs(ur["map50"] - R["validation"]["map50"])))]

    st += [H("4.3 Confidence Threshold 對 Precision / Recall 的取捨", 2)]
    cut = R["map_vs_conf_cutoff"]
    rows = [["送交前丟棄的 confidence 門檻", "剩餘框數", "mAP@0.5"]]
    for s in cut:
        rows.append([fmt(s["conf_cutoff"], 3), str(s["n_boxes"]),
                     fmt(s["map50"])])
    st += [Spacer(1, 2 * mm), table(rows, col_widths=[60 * mm, 40 * mm,
                                                      40 * mm],
                                    highlight_rows=(2,))]
    st += [Spacer(1, 2 * mm),
           figure_block(os.path.join(FIG, "map_vs_cutoff.png"), 130,
                        "圖 5：先過濾低分框會直接傷害 mAP。")]
    st += [P("這就是 stage3_detect.py 把 confidence threshold 固定在 {} 的原因："
             "如果像「給人看」那樣先濾到 0.25，mAP 會從 {:.4f} 掉到 {:.4f}。"
             "產線上真正要給作業員看的門檻（例如 {:.2f}）應該在下游 UI 設定，"
             "而不是在送交給評分器的輸出端先砍掉。"
             .format(R["conf_threshold"], cut[0]["map50"],
                     [s for s in cut if abs(s["conf_cutoff"] - 0.25) < 1e-9][0]["map50"]
                     if any(abs(s["conf_cutoff"] - 0.25) < 1e-9 for s in cut)
                     else cut[-1]["map50"], best["threshold"]))]

    st += [PageBreak(), H("5. Ground Truth vs Prediction 視覺化", 1)]
    st += [P("以下 10 張 validation 影像涵蓋全部六個類別。"
             "上排為 Ground Truth（綠色），下排為 Prediction（紅色，"
             "顯示 confidence ≥ {:.2f} 的框）。".format(R["report_conf"]))]
    st += [figure_block(os.path.join(FIG, "gt_vs_pred.png"), 175,
                        "圖 6：Ground Truth（綠）與 Prediction（紅）對照，"
                        "共 10 張 validation 影像。")]

    st += [PageBreak(), H("6. Failure Analysis", 1)]
    fc = R["failure_counts"]
    st += [table([["失敗型態", "在 validation（conf ≥ {:.2f}）的數量"
                   .format(R["report_conf"])],
                  ["False Negative（完全漏檢）", str(fc.get("false_negative", 0))],
                  ["False Positive（無對應 GT 的誤檢）",
                   str(fc.get("false_positive", 0))],
                  ["Poor Localization（0.1 ≤ IoU < 0.5）",
                   str(fc.get("poor_localization", 0))],
                  ["Class Confusion（位置對但類別錯）",
                   str(fc.get("class_confusion", 0))]],
                 col_widths=[80 * mm, 94 * mm])]
    if os.path.exists(os.path.join(FIG, "failures.png")):
        st += [Spacer(1, 2 * mm),
               figure_block(os.path.join(FIG, "failures.png"), 175,
                            "圖 7：失敗案例（綠 = GT，紅虛線 = Prediction）。")]

    st += [H("6.1 具體案例與原因", 2)]
    cases = R.get("failure_examples", [])[:6]
    rows = [["影像", "型態", "類別", "IoU", "判讀"]]
    reason = {
        "false_negative": "物件太小或與背景紋理對比不足，分數被壓在門檻之下",
        "false_positive": "背景的軋製紋路被誤認為瑕疵，常見於低對比區域",
        "poor_localization": "瑕疵邊界本身模糊，框的範圍判定不一致",
        "class_confusion": "兩類的局部紋理相近，位置抓到但類別判錯",
    }
    for c in cases:
        d = c["detail"]
        cls_idx = int((d.get("gt") or d.get("pred"))[0])
        rows.append([c["image_id"], c["kind"], CLS[cls_idx],
                     fmt(d.get("iou", 0.0), 2), reason.get(c["kind"], "")])
    st += [table(rows, col_widths=[26 * mm, 32 * mm, 28 * mm, 14 * mm,
                                   74 * mm], font_size=7.6)]

    worst = min(CLS, key=lambda c: val["per_class"][c]["ap50"])
    bestc = max(CLS, key=lambda c: val["per_class"][c]["ap50"])
    st += [H("6.2 回答題目指定的五個問題", 2)]
    st += bullets([
        "<b>模型漏掉了哪些 defect？</b>最弱的類別是 {}（AP = {}），"
        "最強的是 {}（AP = {}）。漏檢集中在相對面積小於 3% 的框，"
        "inclusion 有 {:.0f}% 的框落在這個範圍，"
        "在 200×200 的原圖上只有十幾個 pixel 寬，"
        "經過 stride 32 的最深層 feature map 後幾乎沒有空間解析度可用。".format(
            worst, fmt(val["per_class"][worst]["ap50"]), bestc,
            fmt(val["per_class"][bestc]["ap50"]),
            100 * dall["per_class"]["inclusion"]["small_box_ratio"]),
        "<b>Bounding Box 為什麼偏移或過大？</b>crazing 與 rolled_in_scale 是"
        "「整片擴散」的紋理瑕疵，標註者本身就只能畫一個概略範圍；"
        "模型學到的是同樣模糊的邊界，於是 IoU 容易落在 0.3–0.5 之間，"
        "剛好卡在門檻下方被判成 FP。",
        "<b>調整 Confidence Threshold 後 Precision / Recall 如何改變？</b>"
        "見第 4 節：門檻 0.05 → 0.70 時 Precision 由 {:.3f} 升到 {:.3f}，"
        "Recall 由 {:.3f} 降到 {:.3f}。".format(
            sweep[0]["micro_precision"], sweep[-1]["micro_precision"],
            sweep[0]["micro_recall"], sweep[-1]["micro_recall"]),
        "<b>速度是否足以用於產線？</b>單張 {} ms（約 {} FPS）。"
        "典型鋼捲檢測站的節拍遠大於此，單張 GPU 即可即時處理。".format(
            fmt(R["latency_ms_single"], 1),
            fmt(1000.0 / R["latency_ms_single"], 1)),
        "<b>更大的 detector 值得嗎？</b>在這個資料集上，瓶頸是「標註邊界本身模糊」"
        "與「小物件解析度不足」，不是 backbone 容量。"
        "換更大的模型多半只換到個位數百分比的 mAP，卻讓延遲與記憶體成本倍增；"
        "更划算的投資是提高取像解析度與統一標註準則。",
    ])

    st += [PageBreak(), H("7. 部署結論", 1)]
    st += [P("會不會把這個 detector 放到工廠？<b>會，但定位是「輔助標記」而非"
             "「自動判定」。</b>")]
    st += bullets([
        "mAP@0.5 = {}，在六類中有 {} 類 AP 超過 0.8，足以把可疑區域框出來給"
        "作業員複判，大幅縮短人工掃描時間。".format(
            fmt(val["map50"]),
            sum(1 for c in CLS if val["per_class"][c]["ap50"] > 0.8)),
        "不建議直接用來「自動放行／自動報廢」：在 confidence {:.2f} 下 micro "
        "Recall 仍只有 {:.3f}，漏檢的成本（瑕疵流到下游）遠高於誤檢。".format(
            R["report_conf"], val["micro_precision_at_thr"] * 0
            + val["micro_recall_at_thr"]),
        "實務部署建議：產線 UI 用較低門檻（高 Recall）把候選框全部標出，"
        "再由 Stage 2 的分類模型對整張影像做第二層確認，兩者不一致時才送人工。",
        "推論成本 {} ms / 張、checkpoint {} MB，單張入門級 GPU 即可負擔，"
        "不需要額外的加速卡。".format(fmt(R["latency_ms_single"], 1),
                               fmt(R["checkpoint_mb"], 1)),
    ])

    st += [H("8. 繳交內容與執行方式", 1)]
    st += [P("python stage3_detect.py --input_dir &lt;HIDDEN_DIR&gt; "
             "--output_json detections.json", "code")]
    st += bullets([
        "stage3_model.pt 為唯一的 benchmark checkpoint，由 stage3_detect.py "
        "以相對路徑載入，執行期不連網。",
        "輸出每張影像恰好一個 image object；沒有偵測結果時 detections 為 []。",
        "bbox 為原始影像 pixel 座標的 [xmin, ymin, xmax, ymax]，"
        "已 clip 在影像範圍內並排除退化框。",
        "confidence 固定 {}、NMS IoU 固定 {}、每張最多 {} 個框。".format(
            R["conf_threshold"], R["nms_iou_threshold"], R["max_det"]),
    ])

    RB.build(OUT, st, title="FactoryVision Stage 3 — {}".format(TEAM),
             author=TEAM)


if __name__ == "__main__":
    main()
