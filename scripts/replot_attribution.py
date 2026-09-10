#!/usr/bin/env python3
"""Replot the pseudo-forcing attribution figure (paper fig 11) as 1x3.

Panels:
  left:   optimal pseudo-forcing profile (8 local-time 3-hour bins)
  centre: calibration-window day/night LST RMSE over the training subset
  right:  the same attribution bars overlaid with the exchange-air anomaly
          dT that the two-timescale closure produces over the mean diurnal
          cycle, extracted from the continuous year-long integration ---
          the closure's own output follows the diagnosed profile, which is
          the consistency check motivating the closure form.

Run:
  python replot_attribution.py ATTR_DIR DIURNAL_NPZ OUT_PNG
  ATTR_DIR contains structural_history.json + structural_result.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

attr_dir, diurnal_npz, out_png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
history = json.loads((attr_dir / "structural_history.json").read_text())
result = json.loads((attr_dir / "structural_result.json").read_text())
controls = result["final_controls"]
diurnal = np.load(diurnal_npz)
dt_curve = np.asarray(diurnal["dt_air_K"])

bins = np.arange(8) * 3.0 + 1.5
iterations = [h["iteration"] for h in history]


def series(period):
    vals = []
    for h in history:
        entries = [d[period]["rmse_K"] for d in h["days"].values()
                   if d[period]["rmse_K"] is not None]
        vals.append(float(np.sqrt(np.mean(np.square(entries)))) if entries else np.nan)
    return vals


figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)

axes[0].bar(bins, controls[:8], width=2.6, color="tab:blue", alpha=0.7,
            label="attribution $\\Delta T_{air}$ (bins)")
axes[0].plot(bins, dt_curve, color="tab:red", marker="o", markersize=5,
             linewidth=2.0, label="closure $\\Delta T$ (Eq. 4)")
axes[0].axhline(0.0, color="gray", linewidth=0.8)
axes[0].set(xlabel="local hour", ylabel=r"$\Delta T_{air}$ (K)",
            title=f"Pseudo-forcing profile vs closure anomaly\n"
                  f"wind x{controls[8]:.3f}, SW x{controls[9]:.3f}")
axes[0].set_xticks(range(0, 25, 3))
axes[0].legend(fontsize=8, loc="upper left")
axes[0].grid(alpha=0.3)

axes[1].plot(iterations, series("day"), color="tab:red", label="day")
axes[1].plot(iterations, series("night"), color="tab:blue", label="night")
axes[1].set(xlabel="iteration", ylabel="LST RMSE (K)",
            title="Calibration window (training subset)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

figure.suptitle("Pseudo-forcing attribution against MODIS LST", fontsize=11)
figure.savefig(out_png, dpi=260)
print("wrote", out_png)
print("profile:", [round(v, 2) for v in controls[:8]])
print("closure dT:", dt_curve.round(2).tolist())
