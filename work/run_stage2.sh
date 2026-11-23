#!/usr/bin/env bash
# Stage 2 - the full experiment grid.
#   2 architecture families x 2 strategies = the 4 required fair comparisons,
#   plus 3 extra runs used for the deployment analysis.
set -e
cd "$(dirname "$0")/../Team13_Stage2/src"
D="$1"
O="../../work/stage2_runs"
COMMON="--data_root $D --epochs 20 --batch_size 16 --patience 7 --amp --num_workers 4 --out_dir $O"

# ---- required 2 x 2 grid -------------------------------------------------
python -u train.py $COMMON --arch resnet18    --strategy feature_extraction
python -u train.py $COMMON --arch resnet18    --strategy full_finetune
python -u train.py $COMMON --arch densenet121 --strategy feature_extraction
python -u train.py $COMMON --arch densenet121 --strategy full_finetune

# ---- extra strategies / architecture for the deployment discussion -------
python -u train.py $COMMON --arch resnet18    --strategy last_k --last_k 2
python -u train.py $COMMON --arch resnet18    --strategy l2sp
python -u train.py $COMMON --arch mobilenet_v3_small --strategy full_finetune

echo "=== STAGE 2 GRID DONE ==="
