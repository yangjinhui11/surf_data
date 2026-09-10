#!/usr/bin/env python3
"""Plot and summarize the regional CMFD air-temperature sensitivity map."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT = Path(os.environ["SURF_SENSITIVITY_INPUT"])
OUTDIR = Path(os.environ.get("SURF_SENSITIVITY_FIGURE_OUTPUT", INPUT.parent))
OUTLIER_LIMIT = 1.5


def main() -> None:
    data = np.load(INPUT)
    lat, lon = data["lat"], data["lon"]
    skin, sensitivity = data["terminal_skin_t"], data["dskin_dtair"]
    if not np.isfinite(sensitivity).all():
        raise ValueError("sensitivity contains non-finite values")
    outlier = np.abs(sensitivity) > OUTLIER_LIMIT
    stats = {
        "ncolumns": int(sensitivity.size),
        "window_hours": 6.0,
        "median_K_per_K": float(np.median(sensitivity)),
        "p01_K_per_K": float(np.quantile(sensitivity, 0.01)),
        "p05_K_per_K": float(np.quantile(sensitivity, 0.05)),
        "p95_K_per_K": float(np.quantile(sensitivity, 0.95)),
        "p99_K_per_K": float(np.quantile(sensitivity, 0.99)),
        "mean_K_per_K": float(np.mean(sensitivity)),
        "outlier_limit_K_per_K": OUTLIER_LIMIT,
        "outlier_count": int(outlier.sum()),
        "outlier_fraction_percent": float(100.0 * outlier.mean()),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "regional_tair_sensitivity_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10})
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    common = {"s": 2.0, "marker": "s", "linewidths": 0}
    for axis in axes:
        axis.set_xlabel("Longitude (deg E)")
        axis.set_ylabel("Latitude (deg N)")
        axis.set_aspect("equal", adjustable="box")

    plot = axes[0].scatter(lon, lat, c=skin, cmap="turbo", vmin=250, vmax=305, **common)
    axes[0].set_title("Terminal skin temperature")
    figure.colorbar(plot, ax=axes[0], label="K", shrink=0.88)

    clipped = np.clip(sensitivity, 0.0, 1.2)
    plot = axes[1].scatter(lon, lat, c=clipped, cmap="viridis", vmin=0.0, vmax=1.2, **common)
    axes[1].set_title(r"Local $\partial T_{skin}/\partial T_{air}$ (clipped)")
    figure.colorbar(plot, ax=axes[1], label="K K$^{-1}$", shrink=0.88)

    axes[2].scatter(lon, lat, c="0.82", **common)
    axes[2].scatter(lon[outlier], lat[outlier], c="#c43c39", s=9, marker="s", linewidths=0)
    axes[2].set_title(r"Threshold-sensitive columns: $|\partial T_{skin}/\partial T_{air}|>1.5$")
    axes[2].text(0.03, 0.04, f"{stats['outlier_fraction_percent']:.3f}% of columns", transform=axes[2].transAxes)

    figure.savefig(OUTDIR / "regional_tair_sensitivity_maps.png", dpi=300, bbox_inches="tight")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
