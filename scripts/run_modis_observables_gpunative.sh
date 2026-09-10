#!/bin/bash
# Authoritative MODIS-observables integration with the CURRENT GPU-native model.
# Runs run_full_domain_offline.py with SURF_MODIS_TARGETS so each MODIS LST
# pass is sampled from the live integration.  Single authoritative path --
# do not hand-edit per run.
set -euo pipefail
PY=/home/qixiang/.conda/envs/yjh/bin/python
RUNNER=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
SPINUP=/data/yangjinhui/surf_pytorch/surf_paper/experiments/spinup_era5land_2005_2009/restartin_era5land_2009_eq.nc
OBS=/data/yangjinhui/surf_pytorch/surf_paper/observations/modis_cmg/modis_cmg_2018_cmfd_columns.nc
DATA=/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018/cmfd_valid_columns_2018_dataset
CASE=${1:-/data/yangjinhui/surf_pytorch/surf_paper/experiments/prod_gpunative_1y_2018_modis}
GPU=${2:-4}

mkdir -p "$CASE"
ln -sf /data/yangjinhui/surf_pytorch/surf_paper/experiments/debug_1col_32124/metadata.json "$CASE/metadata.json"
ln -sf "$SPINUP" "$CASE/restartin.nc"

cd "$RUNNER"
CUDA_VISIBLE_DEVICES=$GPU \
NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512SKX AVX512CLX AVX2 FMA3" \
SURF_PRELOAD_FORCING=1 \
SURF_CODE=/home/qixiang/yangjinhui/openifs/surf_pytorch \
SURF_DATA="$DATA" \
SURF_CASE="$CASE" \
SURF_MODIS_TARGETS="$OBS" \
SURF_PHYSICS_DT=1800 \
SURF_MAX_STEPS=17520 \
SURF_WRITE_OUTPUT=1 \
SURF_OUTPUT_STRIDE=480 \
"$PY" run_full_domain_offline.py 2>&1 | tee "$CASE/run.log"
echo "MODIS observables written to $CASE/surf_modis_observables.nc"
