#!/usr/bin/env python3
"""Monthly forecast curves: MODIS vs SURF vs SURF-Opt, day and night passes.

Companion to the monthly-RMSE figure (modis_slab_monthly.png): same data
files, same strict-QC masks, same cos(lat) area weights, same nearest-step
view-time sampling protocol, from the continuous year-long integrations.

Run on the server:
  python make_forecast_curves.py OUT_PNG
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

OBS = ("/data/yangjinhui/surf_pytorch/surf_paper/observations/"
       "modis_cmg/modis_cmg_2018_cmfd_columns.nc")
PRIOR = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
         "modis_validation_1y_2018_spunup_20260822/surf_modis_observables.nc")
SLAB = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
        "slab_observables_final_20260824/surf_modis_observables_slab.nc")

MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
CUMULATIVE = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def read_observables(path):
    with Dataset(path) as data:
        return {name: np.asarray(data[name][:], dtype=np.float64)
                for name in ("lst_day", "lst_night")}


def monthly_mean(values, weights, valid, month_of):
    means = np.full(12, np.nan)
    weight_2d = np.broadcast_to(weights[None, :], valid.shape)
    for month in range(1, 13):
        sel = valid & (month_of == month)[:, None]
        if not sel.any():
            continue
        w = weight_2d[sel]
        means[month - 1] = float((values[sel] * w).sum() / w.sum())
    return means


def main():
    out_png = Path(sys.argv[1])
    with Dataset(OBS) as observations:
        obs = {name: np.asarray(observations[name][:], dtype=np.float64)
               for name in ("lst_day", "lst_night")}
        masks = {name: np.asarray(observations[f"{name}_strict"][:]) == 1
                 for name in ("lst_day", "lst_night")}
        lat = np.asarray(observations["lat"][:], dtype=np.float64)
    n_days = obs["lst_day"].shape[0]
    month_of = np.searchsorted(CUMULATIVE, np.arange(1, n_days + 1), side="right") + 1
    weights = np.cos(np.deg2rad(lat))

    prior = read_observables(PRIOR)
    slab = read_observables(SLAB)

    series = {}
    for name in ("lst_day", "lst_night"):
        valid = (masks[name] & np.isfinite(obs[name])
                 & np.isfinite(prior[name]) & np.isfinite(slab[name]))
        series[name] = {
            "MODIS": monthly_mean(obs[name], weights, valid, month_of),
            "SURF": monthly_mean(prior[name], weights, valid, month_of),
            "SURF-Opt": monthly_mean(slab[name], weights, valid, month_of),
        }

    months = np.arange(1, 13)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    for axis, name, title in zip(axes, ("lst_day", "lst_night"),
                                 ("day pass", "night pass")):
        axis.plot(months, series[name]["MODIS"], color="k", marker="o",
                  markersize=4, linewidth=1.6, label="MODIS")
        axis.plot(months, series[name]["SURF"], color="0.55", marker="s",
                  markersize=3.5, linewidth=1.4, label="SURF")
        axis.plot(months, series[name]["SURF-Opt"], color="tab:blue", marker="^",
                  markersize=4, linewidth=1.4, label="SURF-Opt")
        axis.set_xticks(months)
        axis.set_xticklabels(MONTH_LABELS)
        axis.set_ylabel(f"{title} LST (K)", fontsize=9)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
        axis.legend(fontsize=8)
    figure.suptitle("Monthly area-weighted LST: MODIS vs SURF vs SURF-Opt "
                    "(strict QC, continuous 2018 integrations)", fontsize=11)
    figure.savefig(out_png, dpi=260)
    print("wrote", out_png)
    for name in ("lst_day", "lst_night"):
        peak = int(np.nanargmax(series[name]["MODIS"])) + 1
        print(f"{name}: obs peak month {peak}: MODIS {series[name]['MODIS'][peak-1]:.2f} K, "
              f"SURF {series[name]['SURF'][peak-1]:.2f} K, "
              f"SURF-Opt {series[name]['SURF-Opt'][peak-1]:.2f} K")


if __name__ == "__main__":
    main()
