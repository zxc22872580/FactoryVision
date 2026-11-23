"""Stage 3 - build the Ultralytics-style dataset folder from the public split.

The public release already ships YOLO-format labels, but under
``train/labels_yolo`` whereas Ultralytics derives the label path by replacing
``/images/`` with ``/labels/``.  This script materialises

    <out>/images/train, <out>/labels/train
    <out>/images/val,   <out>/labels/val
    <out>/factoryvision.yaml

by copying the files, so the TA-provided split is reproduced exactly and no
re-splitting ever happens.

    python prepare_data.py --data_root <STUDENT_PUBLIC> --out <DIR>
"""
import argparse
import os
import shutil

from common import CLASS_ORDER


def copy_split(src_img, src_lbl, dst_img, dst_lbl):
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(src_img)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        shutil.copyfile(os.path.join(src_img, fn), os.path.join(dst_img, fn))
        lbl = os.path.join(src_lbl, stem + ".txt")
        dst = os.path.join(dst_lbl, stem + ".txt")
        if os.path.exists(lbl):
            shutil.copyfile(lbl, dst)
        else:                     # image without any annotated object
            open(dst, "w").close()
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    det = os.path.join(args.data_root, "detection")
    out = os.path.abspath(args.out)
    counts = {}
    for split, folder in (("train", "train"), ("val", "validation")):
        counts[split] = copy_split(
            os.path.join(det, folder, "images"),
            os.path.join(det, folder, "labels_yolo"),
            os.path.join(out, "images", split),
            os.path.join(out, "labels", split))

    yaml_path = os.path.join(out, "factoryvision.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: {}\n".format(out.replace("\\", "/")))
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for i, c in enumerate(CLASS_ORDER):
            f.write("  {}: {}\n".format(i, c))
    print("train={} val={}\nyaml -> {}".format(counts["train"], counts["val"],
                                               yaml_path))


if __name__ == "__main__":
    main()
