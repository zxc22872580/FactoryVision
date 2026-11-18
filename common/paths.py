"""Shared dataset paths and class definitions for the FactoryVision project."""
import os

CLASS_ORDER = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled_in_scale", "scratches"]
NUM_CLASSES = len(CLASS_ORDER)
SEED = 42

# Dataset root (public release provided by the course).
DATA_ROOT = os.environ.get(
    "FACTORYVISION_DATA",
    r"C:\Users\zxc22\OneDrive\桌面\中興大學\電腦視覺\期末\FactoryVision_STUDENT_PUBLIC\STUDENT_PUBLIC",
)

CLS_TRAIN = os.path.join(DATA_ROOT, "classification", "train")
CLS_VAL = os.path.join(DATA_ROOT, "classification", "validation")
DET_ROOT = os.path.join(DATA_ROOT, "detection")
DET_TRAIN_IMG = os.path.join(DET_ROOT, "train", "images")
DET_TRAIN_XML = os.path.join(DET_ROOT, "train", "annotations_voc")
DET_VAL_IMG = os.path.join(DET_ROOT, "validation", "images")
DET_VAL_XML = os.path.join(DET_ROOT, "validation", "annotations_voc")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(PROJECT_ROOT, "work")
OUTPUTS = os.path.join(WORK, "outputs")
FIGURES = os.path.join(WORK, "figures")
LOGS = os.path.join(WORK, "logs")
for _d in (OUTPUTS, FIGURES, LOGS):
    os.makedirs(_d, exist_ok=True)


def list_classification_split(split_dir):
    """Return (paths, labels) sorted deterministically."""
    paths, labels = [], []
    for idx, cname in enumerate(CLASS_ORDER):
        cdir = os.path.join(split_dir, cname)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                paths.append(os.path.join(cdir, fn))
                labels.append(idx)
    return paths, labels
