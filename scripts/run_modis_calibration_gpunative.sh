#!/bin/bash
# Authoritative MODIS held-out calibration (real + twin), current GPU-native model.
# Protocol per USAGE.md section 2.3.  usage: run_modis_calibration_gpunative.sh <real|twin> [gpu]
set -euo pipefail
MODE=${1:?real|twin}
GPU=${2:-6}
PY=/home/qixiang/.conda/envs/yjh/bin/python
RUNNER=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
SRC=/home/qixiang/yangjinhui/openifs/surf_pytorch
CASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments/prod_gpunative_1y_2018_modis
DATA=/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018/cmfd_valid_columns_2018_dataset
OBS=/data/yangjinhui/surf_pytorch/surf_paper/observations/modis_cmg/modis_cmg_2018_cmfd_columns.nc
OUT=/data/yangjinhui/surf_pytorch/surf_paper/experiments/modis_calibration_gpunative_${MODE}/result.json
mkdir -p "$(dirname "$OUT")"
EXTRA=""
if [ "$MODE" = "twin" ]; then EXTRA="--twin default"; fi
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$RUNNER/modis_calibration_heldout.py" \
  --case "$CASE" --data "$DATA" --observations "$OBS" \
  --state-archive "$CASE/torch_offline_daily.npz" \
  --source "$SRC" --runner "$RUNNER" --output "$OUT" \
  --iterations 30 --learning-rate 0.02 --controls extended $EXTRA \
  --device cuda 2>&1 | tee "$(dirname "$OUT")/run.log"
echo "wrote $(dirname "$OUT")"
