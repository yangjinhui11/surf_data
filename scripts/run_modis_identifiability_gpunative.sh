#!/bin/bash
# Authoritative MODIS parameter identifiability, current GPU-native model.
set -euo pipefail
GPU=${1:-5}
PY=/home/qixiang/.conda/envs/yjh/bin/python
RUNNER=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
SRC=/home/qixiang/yangjinhui/openifs/surf_pytorch
CASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments/prod_gpunative_1y_2018_modis
DATA=/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018/cmfd_valid_columns_2018_dataset
OBS=/data/yangjinhui/surf_pytorch/surf_paper/observations/modis_cmg/modis_cmg_2018_cmfd_columns.nc
OUT=/data/yangjinhui/surf_pytorch/surf_paper/experiments/modis_parameter_identifiability_gpunative
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$RUNNER/modis_parameter_identifiability.py" \
  --case "$CASE" --data "$DATA" --observations "$OBS" \
  --state-archive "$CASE/torch_offline_daily.npz" \
  --source "$SRC" --runner "$RUNNER" --output "$OUT" \
  --sample-days 91,181,271,361 --probes 16 --device cuda 2>&1 | tee "$OUT/run.log"
echo "wrote $OUT/modis_parameter_identifiability.json"
