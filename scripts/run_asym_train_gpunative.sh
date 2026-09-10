#!/bin/bash
# Authoritative two-timescale closure calibration (paper fig modis_structural_slab),
# current GPU-native model.  Protocol per USAGE.md section 2.1: lambda=0.05,
# init tau 10.1 h, training bounds day [6,12] h / night [3,12] h, 40 iterations.
# Single fixed path -- do not hand-edit per run.
set -euo pipefail
GPU=${1:-3}
PY=/home/qixiang/.conda/envs/yjh/bin/python
RUNNER=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
SRC=/home/qixiang/yangjinhui/openifs/surf_pytorch
CASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments/prod_gpunative_1y_2018_modis
DATA=/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018/cmfd_valid_columns_2018_dataset
OBS=/data/yangjinhui/surf_pytorch/surf_paper/observations/modis_cmg/modis_cmg_2018_cmfd_columns.nc
OUT=/data/yangjinhui/surf_pytorch/surf_paper/experiments/asym_train_gpunative
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$RUNNER/modis_slab_asym_train.py" \
  --case "$CASE" --data "$DATA" --observations "$OBS" \
  --state-archive "$CASE/torch_offline_daily.npz" \
  --source "$SRC" --runner "$RUNNER" --output "$OUT" \
  --lambda-value 0.05 --init-tau-hours 10.1 \
  --tau-min-day-hours 6 --tau-min-night-hours 3 --tau-max-hours 12 \
  --iterations 40 --learning-rate 0.05 --device cuda 2>&1 | tee "$OUT/run.log"
echo "wrote $OUT (asym_train_history.json / asym_train_result.json)"
