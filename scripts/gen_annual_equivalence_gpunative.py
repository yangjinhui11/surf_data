# -*- coding: utf-8 -*-
"""Regenerate the paper's one-year autonomous-equivalence figures + metrics
from the GPU-native (pinned, preloaded) full-domain run vs the Fortran
era5land_1y_2018_eq reference.  Writes PNGs to the manuscript figures dir
and a metrics JSON for the LaTeX table."""
import os
os.environ.setdefault("NPY_DISABLE_CPU_FEATURES",
                      "AVX512F AVX512CD AVX512SKX AVX512CLX AVX2 FMA3")
import json
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data/yangjinhui/surf_pytorch/surf_paper/manuscript/figures/annual_repro_gpunative"
os.makedirs(OUT, exist_ok=True)
NPZ = "/data/yangjinhui/surf_pytorch/surf_paper/experiments/prod_gpunative_1y_2018/torch_offline_daily.npz"
F90 = "/data/yangjinhui/surf_pytorch/fortran_runs/era5land_1y_2018_eq/o_gg.nc"
DAT = "/data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset"

lat = np.load(f"{DAT}/lat.npy"); lon = np.load(f"{DAT}/lon.npy")
npz = np.load(NPZ)
steps = np.asarray(npz["step"])

with Dataset(F90) as d:
    ftimestp = np.asarray(d["timestp"][:], dtype=np.int64)
    def fall(name, rec):
        return np.asarray(d.variables[name][rec], dtype=np.float64)

def fseries(name):
    # F90 o_gg: only 2 records (0, 17520); error-growth uses terminal only here.
    return None

# ---- per-checkpoint RMSE evolution (torch npz has 37 records; F90 only terminal) ----
# We can only compare at the terminal (F90 wrote only t=0 and t=17520).
# So error-growth curve uses the torch checkpoints vs F90 terminal for the
# terminal point; for a full curve we'd need F90 10-day outputs.  Here we
# regenerate the *terminal* maps + a metrics table, and note the curve needs
# the F90 10-day reference (era5land_1y_2018_eq_10d) which we also load.
F90_10D = "/data/yangjinhui/surf_pytorch/fortran_runs/era5land_1y_2018_eq_10d/o_gg.nc"
have_10d = os.path.exists(F90_10D)

# resolved-snow mask helper
def snow_mask(swe, rho):
    fsn = np.clip(100.0 * (swe / np.maximum(rho, 1e-12)) * 0.1, 0.0, 1.0)
    return fsn > 1e-3

def rmse(t, f, mask=None):
    m = np.ones(f.shape[0], bool) if mask is None else mask
    d = (t - f)[m]
    return float(np.sqrt((d**2).mean()))

def relrmse(t, f, mask=None):
    m = np.ones(f.shape[0], bool) if mask is None else mask
    return rmse(t, f, m) / float(np.sqrt((f[m]**2).mean())) * 100.0

metrics = {}

# terminal comparison
iT = int(np.flatnonzero(steps == 17520)[0])
with Dataset(F90) as d:
    j = int(np.flatnonzero(ftimestp == 17520)[0])
    swe_f = fall("SWE", j).reshape(-1)
    st_f = fall("SoilTemp", j).reshape(4, -1).T
    sw_f = fall("SoilMoist", j).reshape(4, -1).T
    snt_f = fall("SnowT", j).reshape(-1)
    rho_f = fall("snowdens", j).reshape(-1)
    alb_f = fall("SAlbedo", j).reshape(-1)

swe_t = npz["snow_w"][iT].astype(np.float64).reshape(-1)
st_t = npz["soil_t"][iT].astype(np.float64)
sw_t = npz["soil_w"][iT].astype(np.float64)
snt_t = npz["snow_t"][iT].astype(np.float64).reshape(-1)
rho_t = npz["snow_rho"][iT].astype(np.float64).reshape(-1)
alb_t = npz["snow_alb"][iT].astype(np.float64).reshape(-1)

for L in range(4):
    metrics[f"soil_t_L{L+1}"] = (rmse(st_t[:,L], st_f[:,L]), relrmse(st_t[:,L], st_f[:,L]))
for L in range(4):
    metrics[f"soil_w_L{L+1}"] = (rmse(sw_t[:,L], sw_f[:,L]), relrmse(sw_t[:,L], sw_f[:,L]))
metrics["SWE"] = (rmse(swe_t, swe_f), relrmse(swe_t, swe_f))
m = snow_mask(swe_f, rho_f)
metrics["resolved_snow_cols"] = int(m.sum())
metrics["SnowT"] = (rmse(snt_t, snt_f, m), relrmse(snt_t, snt_f, m))
metrics["snowdens"] = (rmse(rho_t, rho_f, m), relrmse(rho_t, rho_f, m))
metrics["SAlbedo"] = (rmse(alb_t, alb_f, m), relrmse(alb_t, alb_f, m))

with open(os.path.join(OUT, "annual_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ---- error-growth curve (needs F90 10-day reference) ----
if have_10d:
    with Dataset(F90_10D) as d:
        ts10 = np.asarray(d["timestp"][:], dtype=np.int64)
        def f10(name, rec): return np.asarray(d.variables[name][rec], dtype=np.float64)
        # map torch steps to F90 10d records (both are 480-stride)
        common = [s for s in steps if (ts10 == s).any()]
        curves = {"SWE": [], "SoilTemp L1": [], "SnowT": []}
        xs = []
        for s in common:
            ti = int(np.flatnonzero(steps == s)[0]); fj = int(np.flatnonzero(ts10 == s)[0])
            xs.append(s * 1800 / 86400.0)  # days
            sw_f10 = f10("SWE", fj).reshape(-1); rho_f10 = f10("snowdens", fj).reshape(-1)
            curves["SWE"].append(relrmse(npz["snow_w"][ti].reshape(-1), sw_f10))
            curves["SoilTemp L1"].append(relrmse(npz["soil_t"][ti][:,0], f10("SoilTemp", fj).reshape(4,-1).T[:,0]))
            mm = snow_mask(sw_f10, rho_f10)
            curves["SnowT"].append(relrmse(npz["snow_t"][ti].reshape(-1), f10("SnowT", fj).reshape(-1), mm))
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, v in curves.items():
            ax.plot(xs, v, marker="o", ms=3, label=k)
        ax.set_xlabel("day"); ax.set_ylabel("relative RMSE (%)")
        ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "annual_autonomous_error_growth.png"), dpi=200)
        plt.close(fig)

# ---- terminal maps (Fortran / PyTorch / diff) ----
def maps(fields, fname, mmask=None):
    n = len(fields)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.2 * n))
    if n == 1: axes = axes[None, :]
    for r, (label, f90, tch) in enumerate(fields):
        diff = tch - f90
        if mmask is not None:
            f90p = np.where(mmask, f90, np.nan); tchp = np.where(mmask, tch, np.nan)
            dp = np.where(mmask, diff, np.nan)
        else:
            f90p, tchp, dp = f90, tch, diff
        lim = np.nanpercentile(np.abs(dp), 99.5) if np.isfinite(dp).any() else 1.0
        lim = lim if lim > 0 else 1.0
        for c, (arr, ttl) in enumerate([(f90p, "Fortran"), (tchp, "PyTorch (GPU-native)"), (dp, "PyTorch − Fortran")]):
            ax = axes[r, c]
            if c < 2:
                sc = ax.scatter(lon, lat, c=arr, s=1, cmap="viridis")
            else:
                sc = ax.scatter(lon, lat, c=arr, s=1, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"{label} — {ttl}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)

maps([(f"SoilTemp L{L+1} (K)", st_f[:,L], st_t[:,L]) for L in range(4)],
     "annual_autonomous_terminal_soil_temperature_maps.png")
maps([(f"SoilMoist L{L+1} (kg m-2)", sw_f[:,L], sw_t[:,L]) for L in range(4)],
     "annual_autonomous_terminal_soil_water_maps.png")
maps([("SWE (kg m-2)", swe_f, swe_t),
      ("SnowT (K)", snt_f, snt_t),
      ("snowdens (kg m-3)", rho_f, rho_t),
      ("SAlbedo (-)", alb_f, alb_t)],
     "annual_autonomous_terminal_snow_maps.png", mmask=m)

print("figures written to", OUT)
print("metrics:", json.dumps({k: (round(v[0],6), round(v[1],6)) if isinstance(v, tuple) else v for k,v in metrics.items()}, indent=2))
print("10-day reference available:", have_10d)
