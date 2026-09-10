#!/usr/bin/env python3
"""Overlay the closure's exchange-air anomaly on the attribution profile.

Combines the pseudo-forcing attribution bars (the correction the offline
exchange "should" receive, per local 3-hour bin) with the exchange-air
anomaly dT that the two-timescale closure actually produces over the same
local-time axis, extracted from the continuous year-long integration. The
overlay shows the closure form reproduces the shape the attribution
diagnosed.

Run:  python make_attribution_closure_overlay.py ATTR_DIR DIURNAL_NPZ OUT_PNG
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

attr_dir, diurnal_npz, out_png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
result = json.loads((attr_dir / "structural_result.json").read_text())
controls = result["final_controls"]
d = np.load(diurnal_npz)

bins = np.arange(8) * 3.0 + 1.5
attribution = np.asarray(controls[:8])
dt_curve = np.asarray(d["dt_air_K"])

fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
ax.bar(bins, attribution, width=2.6, color="tab:blue", alpha=0.75,
       label="attribution: optimal pseudo-forcing $\\Delta T_{air}$ (3-h bins)")
ax.plot(bins, dt_curve, color="tab:red", marker="o", markersize=6, linewidth=2.2,
        label="closure: exchange-air anomaly $\\Delta T$ (Eq. 4)")
ax.axhline(0.0, color="0.3", linewidth=0.8)
ax.set_xlabel("local hour", fontsize=10)
ax.set_ylabel(r"$\Delta T_{air}$ (K)", fontsize=10)
ax.set_xticks(range(0, 25, 3))
ax.set_title("Attribution diagnosis vs the closure it motivates\n"
             "(domain-mean diurnal cycle, continuous 2018 integration)", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3)
fig.savefig(out_png, dpi=260)
print("wrote", out_png)
print("attribution bars:", attribution.round(2).tolist())
print("closure dT bins :", dt_curve.round(2).tolist())
