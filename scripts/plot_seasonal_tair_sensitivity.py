#!/usr/bin/env python3
"""Plot seasonal regional SURF temperature-response diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_ROOT = Path(os.environ["SURF_SEASONAL_SENSITIVITY_INPUT"])
OUTDIR = Path(os.environ["SURF_SEASONAL_SENSITIVITY_OUTPUT"])
SEASONS = ("spring", "summer", "autumn", "winter")
LABELS = ("Late March", "Late June", "Late September", "Late December")
OUTLIER_LIMIT = 1.5


def main() -> None:
    samples = []
    statistics = []
    for season, label in zip(SEASONS, LABELS):
        data = np.load(INPUT_ROOT / season / "regional_tair_sensitivity.npz")
        sensitivity = data["dskin_dtair"]
        if not np.isfinite(sensitivity).all():
            raise ValueError(f"{season} contains non-finite sensitivity")
        samples.append((data["lat"], data["lon"], sensitivity, label))
        statistics.append({
            "season": season,
            "sample": label,
            "median_K_per_K": float(np.median(sensitivity)),
            "p05_K_per_K": float(np.quantile(sensitivity, 0.05)),
            "p95_K_per_K": float(np.quantile(sensitivity, 0.95)),
            "outlier_fraction_percent": float(100.0 * (np.abs(sensitivity) > OUTLIER_LIMIT).mean()),
        })

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "seasonal_tair_sensitivity_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n"
    )

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10})
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.6), constrained_layout=True)
    scatter = None
    for axis, (lat, lon, sensitivity, label), summary in zip(axes.flat, samples, statistics):
        scatter = axis.scatter(
            lon, lat, c=np.clip(sensitivity, 0.0, 1.2), cmap="viridis",
            vmin=0.0, vmax=1.2, s=1.8, marker="s", linewidths=0,
        )
        threshold_sensitive = np.abs(sensitivity) > OUTLIER_LIMIT
        axis.scatter(
            lon[threshold_sensitive], lat[threshold_sensitive], c="#c43c39",
            s=1.8, marker="s", linewidths=0,
        )
        axis.set_title(label)
        axis.set_xlabel("Longitude (deg E)")
        axis.set_ylabel("Latitude (deg N)")
        axis.set_aspect("equal", adjustable="box")
        axis.text(
            0.02, 0.03,
            f"median {summary['median_K_per_K']:.3f} K/K\n"
            f"|S| > 1.5: {summary['outlier_fraction_percent']:.2f}%",
            transform=axis.transAxes, va="bottom", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
        )
    figure.colorbar(scatter, ax=axes.ravel().tolist(), label=r"$\partial T_{skin}/\partial T_{air}$ (K K$^{-1}$; clipped)", shrink=0.86)
    figure.savefig(OUTDIR / "seasonal_tair_sensitivity_maps.png", dpi=300, bbox_inches="tight")

    figure, axis = plt.subplots(figsize=(6.7, 3.3), constrained_layout=True)
    values = [sample[2][np.abs(sample[2]) <= OUTLIER_LIMIT] for sample in samples]
    box = axis.boxplot(values, labels=LABELS, showfliers=False, patch_artist=True)
    for patch, color in zip(box["boxes"], ("#4c78a8", "#f58518", "#54a24b", "#b279a2")):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axis.set_ylabel(r"$\partial T_{skin}/\partial T_{air}$ (K K$^{-1}$)")
    axis.set_ylim(0.0, 1.2)
    axis.grid(axis="y", color="0.88")
    figure.savefig(OUTDIR / "seasonal_tair_sensitivity_distribution.png", dpi=300, bbox_inches="tight")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
