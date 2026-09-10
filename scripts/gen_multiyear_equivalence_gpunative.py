# -*- coding: utf-8 -*-
"""Multi-panel decadal error-growth figure matching the appendix caption:
top row soil temperature L1-4 + soil water L1-4 (all columns); bottom row
SWE (all) + snow T/density/albedo (resolved-snow mask).  Left: RMSE native
units; right: relative RMSE.  Uses align_multiyear (leap-aware, restartout)."""
import os, sys, json
os.environ.setdefault("NPY_DISABLE_CPU_FEATURES",
                      "AVX512F AVX512CD AVX512SKX AVX512CLX AVX2 FMA3")
sys.path.insert(0, "/tmp")
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import align_multiyear as al

YEARS = list(range(2009, 2019))
TROOT = "/data/yangjinhui/surf_pytorch/surf_paper/experiments/prod_gpunative_10y_2009_2018"
FROOT = "/data/yangjinhui/surf_pytorch/fortran_runs/era5land_multiyear_eq/years"
OUT = "/data/yangjinhui/surf_pytorch/surf_paper/manuscript/figures/multiyear_gpunative"
os.makedirs(OUT, exist_ok=True)

daily = np.load(f"{TROOT}/torch_offline_daily.npz")
steps = np.asarray(daily["step"], dtype=np.int64)
al.selfcheck_year1(daily["snow_w"].astype(np.float64), steps, FROOT, 2009)

def snow_mask(swe, rho):
    return np.clip(100.0 * (swe / np.maximum(rho, 1e-12)) * 0.1, 0.0, 1.0) > 1e-3

# panels: (label, torch_field, f90_field, level, resolved_snow?)
panels = [
    ("Soil temp L1 (K)", "soil_t", "SoilTemp", 0, False),
    ("Soil temp L4 (K)", "soil_t", "SoilTemp", 3, False),
    ("Soil water L1 (kg m$^{-2}$)", "soil_w", "SoilMoist", 0, False),
    ("Soil water L4 (kg m$^{-2}$)", "soil_w", "SoilMoist", 3, False),
    ("SWE (kg m$^{-2}$)", "snow_w", "SWE", None, False),
    ("Snow temp (K)", "snow_t", "SnowT", None, True),
    ("Snow density (kg m$^{-3}$)", "snow_rho", "snowdens", None, True),
    ("Snow albedo (-)", "snow_alb", "SAlbedo", None, True),
]

rmse_all = {p[0]: [] for p in panels}
rel_all = {p[0]: [] for p in panels}
for year in YEARS:
    ti = al.torch_terminal_index(steps, year, YEARS)
    with Dataset(f"{FROOT}/{year}/restartout.nc") as d:
        swe_f = np.asarray(d["SWE"][:], np.float64).reshape(-1)
        rho_f = np.asarray(d["snowdens"][:], np.float64).reshape(-1)
        mask = snow_mask(swe_f, rho_f)
        for label, tn, fn, lev, rs in panels:
            t = daily[tn][ti].astype(np.float64)
            f = np.asarray(d[fn][:], np.float64)
            if lev is not None:
                t = t[:, lev]; f = f.reshape(f.shape[0], -1).T[:, lev]
            else:
                t = t.reshape(-1); f = f.reshape(-1)
            m = mask if rs else None
            tt = t[m] if m is not None else t
            ff = f[m] if m is not None else f
            r = float(np.sqrt(((tt - ff) ** 2).mean()))
            rr = r / float(np.sqrt((ff ** 2).mean())) * 100.0
            rmse_all[label].append(r)
            rel_all[label].append(rr)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# top: soil T & W (all cols); bottom: snow fields
soil = [p for p in panels if not p[4]]
snow = [p for p in panels if p[4] or p[0].startswith("SWE")]
for (ax, data, ylab) in [(axes[0, 0], rmse_all, "RMSE"), (axes[0, 1], rel_all, "relative RMSE (%)")]:
    for label, *_ in soil:
        ax.plot(YEARS, data[label], marker="o", ms=3, label=label)
    ax.set_yscale("log"); ax.set_title("soil (all columns)", fontsize=10)
    ax.set_ylabel(ylab); ax.grid(alpha=0.3); ax.legend(fontsize=6)
for (ax, data, ylab) in [(axes[1, 0], rmse_all, "RMSE"), (axes[1, 1], rel_all, "relative RMSE (%)")]:
    for label, *_ in snow:
        ax.plot(YEARS, data[label], marker="o", ms=3, label=label)
    ax.set_yscale("log"); ax.set_title("snow (SWE all columns; T/$\\rho$/albedo resolved-snow)", fontsize=10)
    ax.set_ylabel(ylab); ax.set_xlabel("year"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
axes[0,0].set_xlabel(""); axes[0,1].set_xlabel("")
fig.suptitle("Decadal Fortran–PyTorch terminal errors, 2009–2018", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "multiyear_annual_terminal_error_growth.png"), dpi=200)
plt.close(fig)
print("multi-panel error-growth figure written")
