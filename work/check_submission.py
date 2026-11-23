# -*- coding: utf-8 -*-
"""Local pre-flight check that mimics what the TA runner does.

    python check_submission.py --stage 1 --submission ../Team13_Stage1 \
        --sample_dir <folder with images>

Checks the required file names, runs the fixed command in a temp directory
(to prove the script does not depend on the current working directory) and
validates the output schema.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

CLASS_ORDER = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled_in_scale", "scratches"]
REQUIRED = {
    1: ["stage1_report.pdf", "stage1_inference.py", "stage1_model.pth",
        "config.json", "requirements.txt", "README.md",
        "src/model.py", "src/train.py", "src/evaluate.py"],
    2: ["stage2_report.pdf", "stage2_inference.py", "stage2_model.pth",
        "config.json", "requirements.txt", "README.md",
        "src/model.py", "src/train.py", "src/evaluate.py"],
    3: ["stage3_report.pdf", "stage3_detect.py", "stage3_model.pt",
        "config.json", "requirements.txt", "README.md",
        "src/model.py", "src/train.py", "src/evaluate.py"],
}
CONFIG_FIELDS = {
    1: ["team_id", "stage", "model_name", "input_size", "class_order",
        "normalization_mean", "normalization_std", "seed",
        "improvement_methods", "checkpoint_file"],
    2: ["team_id", "stage", "architecture", "strategy", "pretrained_source",
        "input_size", "class_order", "normalization_mean",
        "normalization_std", "seed", "checkpoint_file"],
    3: ["team_id", "stage", "detector", "pretrained_source", "input_size",
        "class_order", "confidence_threshold", "nms_iou_threshold",
        "bbox_format", "seed", "checkpoint_file"],
}

ok_n, fail = 0, []


def check(cond, msg):
    global ok_n
    if cond:
        ok_n += 1
        print("  [OK]   " + msg)
    else:
        fail.append(msg)
        print("  [FAIL] " + msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--submission", required=True)
    ap.add_argument("--sample_dir", required=True)
    ap.add_argument("--skip_run", action="store_true")
    args = ap.parse_args()
    sub = os.path.abspath(args.submission)
    stage = args.stage

    print("== file structure ==")
    expected_root = "Team13_Stage{}".format(stage)
    check(os.path.basename(sub).endswith("_Stage{}".format(stage)),
          "root folder name ends with _Stage{} (got '{}')".format(
              stage, os.path.basename(sub)))
    for rel in REQUIRED[stage]:
        check(os.path.exists(os.path.join(sub, rel)), "exists: " + rel)
    for extra in os.listdir(sub):
        if extra not in [r.split("/")[0] for r in REQUIRED[stage]]:
            print("  [note] extra entry in root: " + extra)

    print("== config.json ==")
    try:
        cfg = json.load(open(os.path.join(sub, "config.json"),
                             encoding="utf-8"))
        for f in CONFIG_FIELDS[stage]:
            check(f in cfg, "config field: " + f)
        check(cfg.get("class_order") == CLASS_ORDER,
              "class_order matches the fixed order")
        ck = cfg.get("checkpoint_file")
        check(ck and os.path.exists(os.path.join(sub, ck)),
              "checkpoint_file '{}' exists".format(ck))
        if stage == 3:
            check(cfg.get("bbox_format") == "xyxy_pixel",
                  "bbox_format == 'xyxy_pixel'")
    except Exception as e:                                  # noqa: BLE001
        check(False, "config.json parses: " + str(e))

    print("== requirements.txt ==")
    req = open(os.path.join(sub, "requirements.txt"),
               encoding="utf-8").read().splitlines()
    bad = [ln for ln in req if ln.strip() and
           any(t in ln for t in ("http://", "https://", "git+", "-e ",
                                 "file:", "C:\\", "/home/", "/Users/"))]
    check(not bad, "no URL / editable / local-path requirement ({})".format(
        bad or "clean"))

    print("== no personal absolute paths in shipped code ==")
    offenders = []
    for root, _d, files in os.walk(sub):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            txt = open(p, encoding="utf-8", errors="ignore").read()
            for tok in ("C:\\Users", "C:/Users", "/home/", "/Users/"):
                if tok in txt:
                    offenders.append(os.path.relpath(p, sub))
                    break
    check(not offenders, "no hard-coded personal path ({})".format(
        offenders or "clean"))

    if args.skip_run:
        return summary()

    print("== run the fixed command from a different working directory ==")
    tmp = tempfile.mkdtemp(prefix="favcheck_")
    try:
        if stage == 3:
            out = os.path.join(tmp, "detections.json")
            cmd = [sys.executable, os.path.join(sub, "stage3_detect.py"),
                   "--input_dir", os.path.abspath(args.sample_dir),
                   "--output_json", out]
        else:
            out = os.path.join(tmp, "predictions.csv")
            cmd = [sys.executable,
                   os.path.join(sub, "stage{}_inference.py".format(stage)),
                   "--input_dir", os.path.abspath(args.sample_dir),
                   "--output_csv", out]
        print("  $ " + " ".join(cmd))
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        check(r.returncode == 0, "return code == 0")
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-3000:])
            return summary()
        check(os.path.exists(out), "output file created")

        imgs = [f for f in sorted(os.listdir(args.sample_dir))
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        stems = [os.path.splitext(f)[0] for f in imgs]
        if stage == 3:
            data = json.load(open(out, encoding="utf-8"))
            check(isinstance(data, list), "output is a JSON array")
            check(len(data) == len(stems),
                  "exactly one object per image ({}/{})".format(len(data),
                                                                len(stems)))
            check(sorted(d["image_id"] for d in data) == sorted(stems),
                  "image_id set matches the filename stems")
            okbox, okconf, okcls, okmax = True, True, True, True
            from PIL import Image
            for d in data:
                if len(d["detections"]) > 100:
                    okmax = False
                w, h = Image.open(os.path.join(args.sample_dir,
                                               d["image_id"] + ".png")).size
                for det in d["detections"]:
                    x1, y1, x2, y2 = det["bbox"]
                    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                        okbox = False
                    c = det["confidence"]
                    if not (0.0 <= c <= 1.0) or c != c:
                        okconf = False
                    if det["class"] not in CLASS_ORDER:
                        okcls = False
            check(okbox, "every bbox satisfies 0<=x1<x2<=W, 0<=y1<y2<=H")
            check(okconf, "every confidence in [0,1], no NaN/Inf")
            check(okcls, "every class is one of the six names")
            check(okmax, "at most 100 detections per image")
            print("  [info] total detections: {}".format(
                sum(len(d["detections"]) for d in data)))
        else:
            rows = list(csv.reader(open(out, encoding="utf-8")))
            check(rows[0] == ["image_id", "prediction"],
                  "header is exactly image_id,prediction")
            body = rows[1:]
            check(len(body) == len(stems),
                  "one row per image ({}/{})".format(len(body), len(stems)))
            ids = [r[0] for r in body]
            check(sorted(ids) == sorted(stems),
                  "image_id set matches the filename stems")
            check(len(set(ids)) == len(ids), "no duplicated image_id")
            check(all(len(r) == 2 for r in body), "exactly two columns")
            bad = sorted({r[1] for r in body} - set(CLASS_ORDER))
            check(not bad, "predictions use only the six class names ({})"
                  .format(bad or "clean"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return summary()


def summary():
    print("\n==== {} checks passed, {} failed ====".format(ok_n, len(fail)))
    for f in fail:
        print("  - " + f)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
