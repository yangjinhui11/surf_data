#!/usr/bin/env python3
"""Replot the slab maps figure (paper fig 12) from the archived slab_maps.npz.

Replaces the low-information 2x2 (window x model) layout with a single row of
three high-information panels for the held-out spring-summer window:
  (a) SURF column-mean daytime bias
  (b) SURF-Opt column-mean daytime bias
  (c) |bias| change = |SURF| - |SURF-Opt|  (positive = the closure reduces the
      absolute bias, negative = it increases it)

Run:  python replot_slab_maps.py SLAB_MAPS_NPZ OUT_PNG
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

npz_path, out_png = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(npz_path)
lat, lon = d["lat"], d["lon"]
surf = d["lambda_0.0000__validation__day"]
opt = d["lambda_0.0500__validation__day"]
count = d["lambda_0.0000__validation__day__count"]
mask = (count > 0) & np.isfinite(surf) & np.isfinite(opt)
weights = np.cos(np.deg2rad(lat[mask]))


def wmean(x):
    return float((x[mask] * weights).sum() / weights.sum())


improvement = np.abs(surf) - np.abs(opt)  # positive => closure reduces |bias|
improvement[~mask] = np.nan

figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)

sc0 = axes[0].scatter(lon[mask], lat[mask], c=surf[mask], cmap="RdBu_r",
                      vmin=-12, vmax=12, s=0.25, linewidths=0)
axes[0].set_title(f"SURF: column-mean day bias (mean {wmean(surf):+.2f} K)", fontsize=10)

sc1 = axes[1].scatter(lon[mask], lat[mask], c=opt[mask], cmap="RdBu_r",
                      vmin=-12, vmax=12, s=0.25, linewidths=0)
axes[1].set_title(f"SURF-Opt: column-mean day bias (mean {wmean(opt):+.2f} K)", fontsize=10)

sc2 = axes[2].scatter(lon[mask], lat[mask], c=improvement[mask], cmap="RdYlGn",
                      vmin=-6, vmax=6, s=0.25, linewidths=0)
frac = float((improvement[mask] > 0).mean() * 100)
axes[2].set_title("|bias| change, SURF-Opt minus SURF\n(green = the closure reduces the absolute bias)",
                  fontsize=10)

for ax in axes:
    ax.set_xlabel("lon", fontsize=8)
    ax.set_ylabel("lat", fontsize=8)
    ax.tick_params(labelsize=7)

figure.colorbar(sc0, ax=axes[:2].tolist(), label="daytime bias, model - MODIS (K)",
                shrink=0.85, location="bottom")
figure.colorbar(sc2, ax=axes[2], label="|bias| change (K)", shrink=0.85, location="bottom")

figure.suptitle("Daytime LST bias vs MODIS, held-out spring-summer window: SURF vs SURF-Opt "
                "(two-timescale surface-layer closure, "
                "lambda_s=0.05 K/(W m$^{-2}$), tau_day=6 h, tau_night=3 h)", fontsize=11)
figure.savefig(out_png, dpi=260)
print("wrote", out_png)
print(f"SURF mean {wmean(surf):+.2f} K; SURF-Opt mean {wmean(opt):+.2f} K; "
      f"improved columns {frac:.1f}%")
