#!/usr/bin/env python3
"""Two unified seasonal-cycle figures from the MODIS observables files.

Figure A: observation-versus-forecast monthly curves for the prior
  (day-pass LST, night-pass LST, snow fraction), area weighted, obs QC
  masks applied to both curves.
Figure B: monthly LST RMSE, prior versus slab closure, day and night passes,
  computed from the prior and slab observable files under identical
  protocols (continuous full-year integrations, nearest-step view-time
  sampling).

Also writes seasonal_cycle_stats.json with every number used in captions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from netCDF4 import Dataset  # noqa: E402

OBS = ("/data/yangjinhui/surf_pytorch/surf_paper/observations/"
       "modis_cmg/modis_cmg_2018_cmfd_columns.nc")
PRIOR = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
         "modis_validation_1y_2018_spunup_20260822/surf_modis_observables.nc")
SLAB = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
        "slab_observables_20260824/surf_modis_observables_slab.nc")
OUT = Path("/data/yangjinhui/surf_pytorch/surf_paper/experiments/slab_observables_20260824")

MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
CUMULATIVE = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def read_observables(path: str) -> dict:
    with Dataset(path) as data:
        return {name: np.asarray(data[name][:], dtype=np.float64) for name in
                ("lst_day", "lst_night", "snow_fraction")}


def monthly_means(residual_basis: np.ndarray, weights_1d: np.ndarray,
                  valid: np.ndarray, month_of: np.ndarray, n_obs_days: int) -> dict:
    """Area-weighted monthly mean of `residual_basis` over `valid` samples."""
    sums = np.zeros(12)
    weight_sums = np.zeros(12)
    counts = np.zeros(12, dtype=int)
    weight_2d = np.broadcast_to(weights_1d[None, :], valid.shape)
    for month in range(1, 13):
        selected = valid & (month_of == month)[:, None]
        if not selected.any():
            continue
        w = weight_2d[selected]
        sums[month - 1] = float((residual_basis[selected] * w).sum())
        weight_sums[month - 1] = float(w.sum())
        counts[month - 1] = int(selected.sum())
    mean = np.full(12, np.nan)
    has = weight_sums > 0
    mean[has] = sums[has] / weight_sums[has]
    return {"mean": mean, "count": counts}


def main() -> None:
    with Dataset(OBS) as observations:
        obs = {name: np.asarray(observations[name][:], dtype=np.float64) for name in
               ("lst_day", "lst_night", "snow_fraction")}
        masks = {
            "lst_day": np.asarray(observations["lst_day_strict"][:]) == 1,
            "lst_night": np.asarray(observations["lst_night_strict"][:]) == 1,
            "snow_fraction": np.asarray(observations["snow_good"][:]) == 1,
        }
        lat = np.asarray(observations["lat"][:], dtype=np.float64)
    n_obs_days = obs["lst_day"].shape[0]
    month_of = (np.searchsorted(CUMULATIVE, np.arange(1, n_obs_days + 1), side="right") + 1)
    weights = np.cos(np.deg2rad(lat))

    prior = read_observables(PRIOR)
    slab = read_observables(SLAB)

    stats: dict = {}
    valid_pairs: dict = {}
    for name in ("lst_day", "lst_night", "snow_fraction"):
        valid = masks[name] & np.isfinite(obs[name]) & np.isfinite(prior[name]) & np.isfinite(slab[name])
        valid_pairs[name] = valid
        stats[name] = {
            "obs_mean": monthly_means(obs[name], weights, valid, month_of, n_obs_days),
            "prior_mean": monthly_means(prior[name], weights, valid, month_of, n_obs_days),
            "slab_mean": monthly_means(slab[name], weights, valid, month_of, n_obs_days),
            "prior_rmse": monthly_means((prior[name] - obs[name]) ** 2, weights, valid, month_of, n_obs_days),
            "slab_rmse": monthly_means((slab[name] - obs[name]) ** 2, weights, valid, month_of, n_obs_days),
        }
        for entry, values in stats[name].items():
            if entry.endswith("rmse"):
                values["mean"] = np.sqrt(values["mean"])

    months = np.arange(1, 13)

    # ------------------------------------------------------------------
    # Figure A: observation vs prior forecast, monthly means.
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    panels = (
        ("lst_day", "day-pass LST (K)"),
        ("lst_night", "night-pass LST (K)"),
        ("snow_fraction", "snow fraction (-)"),
    )
    for axis, (name, ylabel) in zip(axes, panels):
        axis.plot(months, stats[name]["obs_mean"]["mean"], color="k", marker="o",
                  markersize=4, linewidth=1.4, label="MODIS")
        axis.plot(months, stats[name]["prior_mean"]["mean"], color="tab:red",
                  marker="s", markersize=3.5, linewidth=1.4, label="SURF")
        axis.set_xticks(months)
        axis.set_xticklabels(MONTH_LABELS)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
        if name == "snow_fraction":
            axis.set_ylim(0.0, max(0.6, float(np.nanmax(stats[name]["obs_mean"]["mean"])) * 1.15))
    axes[0].legend(fontsize=8, loc="upper left")
    figure.suptitle("Monthly area-weighted means: MODIS observations vs the offline "
                    "SURF forecast (strict QC, 2018)", fontsize=11)
    figure.savefig(OUT / "modis_seasonal_cycle.png", dpi=260)
    plt.close(figure)

    # ------------------------------------------------------------------
    # Figure B: monthly RMSE, prior vs slab closure.
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    for axis, name, title in zip(axes, ("lst_day", "lst_night"),
                                 ("day pass", "night pass")):
        axis.plot(months, stats[name]["prior_rmse"]["mean"], color="0.55",
                  marker="o", markersize=4, linewidth=1.4, label="SURF")
        axis.plot(months, stats[name]["slab_rmse"]["mean"], color="tab:blue",
                  marker="s", markersize=3.5, linewidth=1.4, label="SURF-Opt")
        axis.set_xticks(months)
        axis.set_xticklabels(MONTH_LABELS)
        axis.set_ylabel(f"{title} RMSE (K)", fontsize=9)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
        axis.legend(fontsize=8)
    figure.suptitle("Monthly area-weighted LST RMSE against MODIS: SURF vs "
                    "SURF-Opt (2018)", fontsize=11)
    figure.savefig(OUT / "modis_slab_monthly.png", dpi=260)
    plt.close(figure)

    # ------------------------------------------------------------------
    # Caption facts.
    # ------------------------------------------------------------------
    def amplitude(series: np.ndarray) -> float:
        return float(np.nanmax(series) - np.nanmin(series))

    facts = {
        "counts": {name: int(valid_pairs[name].sum()) for name in valid_pairs},
        "seasonal_amplitude": {
            "lst_day_obs": amplitude(stats["lst_day"]["obs_mean"]["mean"]),
            "lst_day_prior": amplitude(stats["lst_day"]["prior_mean"]["mean"]),
            "lst_night_obs": amplitude(stats["lst_night"]["obs_mean"]["mean"]),
            "lst_night_prior": amplitude(stats["lst_night"]["prior_mean"]["mean"]),
            "snow_obs": amplitude(stats["snow_fraction"]["obs_mean"]["mean"]),
            "snow_prior": amplitude(stats["snow_fraction"]["prior_mean"]["mean"]),
        },
        "snow_max_obs": float(np.nanmax(stats["snow_fraction"]["obs_mean"]["mean"])),
        "snow_max_prior": float(np.nanmax(stats["snow_fraction"]["prior_mean"]["mean"])),
        "monthly_series": {
            name: {entry: values["mean"].tolist() for entry, values in stats[name].items()}
            for name in stats
        },
    }
    day_prior = stats["lst_day"]["prior_rmse"]["mean"]
    day_slab = stats["lst_day"]["slab_rmse"]["mean"]
    night_prior = stats["lst_night"]["prior_rmse"]["mean"]
    night_slab = stats["lst_night"]["slab_rmse"]["mean"]
    facts["day_rmse_reduction_K"] = {
        "min": float(np.nanmin(day_prior - day_slab)),
        "max": float(np.nanmax(day_prior - day_slab)),
        "months_improved": int(np.nansum(day_slab < day_prior)),
    }
    facts["night_rmse_change_K"] = {
        "min": float(np.nanmin(night_slab - night_prior)),
        "max": float(np.nanmax(night_slab - night_prior)),
        "months_degraded": int(np.nansum(night_slab > night_prior)),
    }
    (OUT / "seasonal_cycle_stats.json").write_text(json.dumps(facts, indent=1) + "\n")
    print(json.dumps({key: facts[key] for key in
                      ("counts", "seasonal_amplitude", "snow_max_obs", "snow_max_prior",
                       "day_rmse_reduction_K", "night_rmse_change_K")}, indent=1))


if __name__ == "__main__":
    main()
