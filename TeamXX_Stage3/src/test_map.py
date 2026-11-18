# -*- coding: utf-8 -*-
"""Unit tests for the mAP@0.5 implementation in evaluate.py.

The official grade is computed by the TA runner, so our own metric has to
reproduce its definition exactly.  Each test below pins down one clause of
that definition.

    python test_map.py
"""
import numpy as np

from evaluate import evaluate_detections, iou_matrix, voc_ap_all_points

PASS, FAIL = [], []


def check(name, got, want, tol=1e-4):
    ok = abs(got - want) <= tol
    (PASS if ok else FAIL).append(name)
    print("{:5s} {:52s} got {:.4f}  want {:.4f}".format(
        "OK" if ok else "FAIL", name, got, want))


def main():
    gt = {"a": {"boxes": [(0, 10, 10, 50, 50)]},
          "b": {"boxes": [(0, 0, 0, 20, 20)]}}

    # 1. a perfect detector scores AP = 1 for that class, and mAP averages
    #    over all six classes (the five empty ones contribute AP = 0).
    r = evaluate_detections(gt, {"a": [(0, 0.9, 10, 10, 50, 50)],
                                 "b": [(0, 0.8, 0, 0, 20, 20)]})
    check("perfect detection -> AP = 1", r["per_class"]["crazing"]["ap50"], 1.0)
    check("mAP averages over all 6 classes", r["map50"], 1.0 / 6)

    # 2. a second box on an already-matched GT is a False Positive.
    r = evaluate_detections(gt, {"a": [(0, 0.9, 10, 10, 50, 50),
                                       (0, 0.8, 11, 11, 50, 50)],
                                 "b": [(0, 0.7, 0, 0, 20, 20)]})
    check("duplicate box on the same GT counts as FP",
          r["per_class"]["crazing"]["ap50"], 0.8333)

    # 3. the right box with the wrong class is a FP, and the GT stays missed.
    r = evaluate_detections(gt, {"a": [(1, 0.9, 10, 10, 50, 50)],
                                 "b": [(0, 0.8, 0, 0, 20, 20)]})
    check("wrong class -> FP for that class",
          r["per_class"]["inclusion"]["ap50"], 0.0)
    check("wrong class -> the GT remains unmatched",
          r["per_class"]["crazing"]["ap50"], 0.5)

    # 4. the IoU threshold is exactly 0.5.
    gt4 = {"x": {"boxes": [(0, 0, 0, 100, 100)]}}
    for shift, want in ((34, 0.0), (32, 1.0)):
        iou = iou_matrix(np.array([[shift, 0, 100 + shift, 100]]),
                         np.array([[0, 0, 100, 100]]))[0, 0]
        r = evaluate_detections(
            gt4, {"x": [(0, 0.9, shift, 0, 100 + shift, 100)]})
        check("IoU {:.3f} {} 0.5".format(iou, "<" if want == 0 else ">"),
              r["per_class"]["crazing"]["ap50"], want)

    # 5. Pascal VOC 2010+ all-points interpolation.
    rec = np.array([0.2, 0.4, 0.4, 0.6, 0.8, 1.0])
    pre = np.array([1.0, 1.0, 0.6, 0.7, 0.6, 0.5])
    check("VOC2010+ all-points interpolation",
          voc_ap_all_points(rec, pre), 0.76)

    # 6. a prediction on an image that has no GT of that class is a FP.
    r = evaluate_detections({"x": {"boxes": [(0, 0, 0, 100, 100)]}},
                            {"x": [(0, 0.9, 0, 0, 100, 100)],
                             "y": [(0, 0.95, 0, 0, 50, 50)]})
    check("prediction without any GT counts as FP",
          r["per_class"]["crazing"]["ap50"], 0.5)

    print("\n{} passed, {} failed".format(len(PASS), len(FAIL)))
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
