#!/usr/bin/env bash
# Stage 3 - prepare the Ultralytics dataset folder and fine-tune YOLOv8s.
#
#   bash run_stage3.sh <STUDENT_PUBLIC> [epochs] [batch] [imgsz]
#
# Absolute paths are used for --project because Ultralytics resolves a
# relative project path against its own settings directory, not the cwd.
set -e
cd "$(dirname "$0")/../Team13_Stage3/src"
D="$1"
WORK="$(cd ../../work && pwd)"
DS="$WORK/det_dataset"
EPOCHS="${2:-150}"
BATCH="${3:-16}"
IMGSZ="${4:-640}"

python -u prepare_data.py --data_root "$D" --out "$DS"
python -u train.py \
    --data_yaml "$DS/factoryvision.yaml" \
    --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ" \
    --pretrained "$WORK/yolov8s.pt" \
    --project "$WORK/stage3_runs" --name "yolov8s_${IMGSZ}" \
    --workers 8

echo "=== STAGE 3 TRAINING DONE ==="
