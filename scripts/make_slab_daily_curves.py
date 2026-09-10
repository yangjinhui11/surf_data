# -*- coding: utf-8 -*-
"""Paper figure modis_slab_daily_curves.png: daily area-weighted mean LST,
MODIS (black) vs SURF (grey) vs SURF-Opt (blue), day and night passes, with a
7-day running mean.  Authoritative producer; paths via env with the current
GPU-native products as defaults."""
import os
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBS = os.environ.get(
    "SURF_MODIS_OBS",
    "/data/yangjinhui/surf_pytorch/surf_paper/observations/modis_cmg/modis_cmg_2018_cmfd_columns.nc")
PRIOR = os.environ.get(
    "SURF_PRIOR_OBS",
    "/data/yangjinhui/surf_pytorch/surf_paper/experiments/prod_gpunative_1y_2018_modis/surf_modis_observables.nc")
SLAB = os.environ.get(
    "SURF_SLAB_OBS",
    "/data/yangjinhui/surf_pytorch/surf_paper/experiments/slab_observables_gpunative/surf_modis_observables_slab.nc")
OUT = os.environ.get(
    "SURF_SLAB_FIG_OUT",
    "/data/yangjinhui/surf_pytorch/surf_paper/experiments/slab_figures_gpunative")
os.makedirs(OUT, exist_ok=True)


def read(path, fields):
    with Dataset(path) as d:
        return {f: np.asarray(d[f][:], dtype=np.float64) for f in fields}, \
               np.asarray(d["lat"][:], dtype=np.float64)


def daily_mean(field, lat, valid):
    w = np.cos(np.deg2rad(lat))
    w2 = np.broadcast_to(w[None, :], field.shape)
    num = np.where(valid, field * w2, 0.0).sum(axis=1)
    den = np.where(valid, w2, 0.0).sum(axis=1)
    out = np.full(field.shape[0], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


obs, lat = read(OBS, ("lst_day", "lst_night"))
strict = {"lst_day": "lst_day_strict", "lst_night": "lst_night_strict"}
with Dataset(OBS) as d:
    masks = {k: np.asarray(d[v][:]) == 1 for k, v in strict.items()}
prior, _ = read(PRIOR, ("lst_day", "lst_night"))
slab, _ = read(SLAB, ("lst_day", "lst_night"))

def runmean(x, w=7):
    k = np.ones(w) / w
    c = np.convolve(np.nan_to_num(x, nan=0.0), k, mode="same")
    n = np.convolve(np.isfinite(x).astype(float), k, mode="same")
    out = np.full_like(x, np.nan)
    ok = n > 0
    out[ok] = c[ok] / n[ok]
    return out

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
days = np.arange(1, obs["lst_day"].shape[0] + 1)
for ax, name, ttl in [(axes[0], "lst_day", "day pass"), (axes[1], "lst_night", "night pass")]:
    m = masks[name]
    o = daily_mean(obs[name], lat, m)
    p = daily_mean(prior[name], lat, m)
    s = daily_mean(slab[name], lat, m)
    ax.plot(days, o, color="k", lw=0.4, alpha=0.4)
    ax.plot(days, p, color="0.5", lw=0.4, alpha=0.4)
    ax.plot(days, s, color="tab:blue", lw=0.4, alpha=0.4)
    ax.plot(days, runmean(o), color="k", lw=1.6, label="MODIS")
    ax.plot(days, runmean(p), color="0.5", lw=1.6, label="SURF")
    ax.plot(days, runmean(s), color="tab:blue", lw=1.6, label="SURF-Opt")
    ax.set_ylabel("LST (K)"); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, ncol=3)
axes[1].set_xlabel("day of 2018")
fig.suptitle("Daily area-weighted mean LST (strict QC): MODIS vs SURF vs SURF-Opt", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "modis_slab_daily_curves.png"), dpi=260)
print("wrote", os.path.join(OUT, "modis_slab_daily_curves.png"))
