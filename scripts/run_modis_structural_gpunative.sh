#!/bin/bash
# Authoritative MODIS structural-closure driver (attribution + slab), current
# GPU-native model.  Single fixed path; do not hand-edit per run.
#   usage: run_modis_structural_gpunative.sh <attribution|slab> [gpu]
set -euo pipefail
MODE=${1:?attribution|slab}
GPU=${2:-4}
PY=/home/qixiang/.conda/envs/yjh/bin/python
RUNNER=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
SRC=/home/qixiang/yangjinhui/openifs/surf_pytorch
# latest GPU-native 2018 run (provides metadata + daily state archive)
CASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments/prod_gpunative_1y_2018_modis
DATA=/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018/cmfd_valid_columns_2018_dataset
OBS=/data/yangjinhui/surf_pytorch/surf_paper/observations/modis_cmg/modis_cmg_2018_cmfd_columns.nc
OUT=/data/yangjinhui/surf_pytorch/surf_paper/experiments/structural_closure_gpunative_${MODE}

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$RUNNER/modis_structural_closure.py" \
  --mode "$MODE" \
  --case "$CASE" --data "$DATA" --observations "$OBS" \
  --state-archive "$CASE/torch_offline_daily.npz" \
  --source "$SRC" --runner "$RUNNER" --output "$OUT" \
  --device cuda 2>&1 | tee "$OUT/run.log"
echo "wrote $OUT"
