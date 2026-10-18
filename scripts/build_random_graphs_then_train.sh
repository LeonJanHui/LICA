#!/usr/bin/env bash
set -euo pipefail
PY=/home/dldx/miniconda3/envs/boltz-egnn/bin/python
ROOT=/home/dldx/Boltz-EGNN-Affinity
LOG=$ROOT/results/random_split/train.log
mkdir -p "$(dirname "$LOG")"

echo "[START] $(date)" | tee -a "$LOG"
cd "$ROOT/src"

echo "[BUILD] train/valid/casf graphs" | tee -a "$LOG"
$PY -u "$ROOT/scripts/build_split_graphs.py" --config "$ROOT/configs/train_config.yaml" \
  2>&1 | tee -a "$LOG"

echo "[TRAIN] $(date)" | tee -a "$LOG"
$PY -u "$ROOT/src/train.py" --config "$ROOT/configs/train_config.yaml" \
  2>&1 | tee -a "$LOG"

echo "[FINISH] $(date)" | tee -a "$LOG"
