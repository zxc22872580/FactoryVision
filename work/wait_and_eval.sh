#!/usr/bin/env bash
# Wait for the Stage 3 training to finish, then run the full validation
# evaluation + visualisation in one go.
cd "$(dirname "$0")"
LOG=logs/stage3.log
W=stage3_runs/yolov8s_640/weights/best.pt
for i in $(seq 1 480); do          # at most 4 hours
    if grep -q "STAGE 3 TRAINING DONE" "$LOG" 2>/dev/null; then
        echo "training finished after ~$((i/2)) minutes of waiting"
        break
    fi
    python -c "import time;time.sleep(30)"
done
if ! grep -q "STAGE 3 TRAINING DONE" "$LOG" 2>/dev/null; then
    echo "TIMED OUT waiting for training"; exit 1
fi
test -f "$W" || { echo "missing $W"; exit 1; }
cp stage3_runs/yolov8s_640/train_config.json outputs/stage3_train_config.json
python -u run_stage3_eval.py --weights "$W" --data_root "$1" --imgsz 640 --data_yaml "$(pwd)/det_dataset/factoryvision.yaml"
echo "=== STAGE 3 EVAL DONE ==="
