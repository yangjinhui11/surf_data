#!/bin/bash
# Authoritative slab per-day/maps evaluation, current GPU-native model.
# Protocol per paper: lambda in {0.0, 0.05}, tau (day,night)=(6,3) h, slab
# depth 600 m, calibration/validation windows per USAGE.md.
set -euo pipefail
GPU=${1:-6}
PY=/home/qixiang/.conda/envs/yjh/bin/python
RUNNER=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
SRC=/home/qixiang/yangjinhui/openifs/surf_pytorch
CASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments/prod_gpunative_1y_2018_modis
DATA=/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018/cmfd_valid_columns_2018_dataset
OBS=/data/yangjinhui/surf_pytorch/surf_paper/observations/modis_cmg/modis_cmg_2018_cmfd_columns.nc
OUT=/data/yangjinhui/surf_pytorch/surf_paper/experiments/slab_maps_gpunative/slab_maps.npz
mkdir -p "$(dirname "$OUT")"
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$RUNNER/modis_slab_maps.py" \
  --case "$CASE" --data "$DATA" --observations "$OBS" \
  --state-archive "$CASE/torch_offline_daily.npz" \
  --source "$SRC" --runner "$RUNNER" --output "$OUT" \
  --lambda-values 0.0,0.05 --tau-day-hours 6 --tau-night-hours 3 \
  --slab-depth 600.0 --device cuda 2>&1 | tee "$(dirname "$OUT")/run.log"
echo "wrote $(dirname "$OUT")/slab_maps.png and slab_perday.png"
