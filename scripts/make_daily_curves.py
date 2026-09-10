#!/usr/bin/env python3
"""Daily LST curves: MODIS vs SURF vs SURF-Opt, day and night passes.

Daily-resolution companion to the monthly forecast-curves figure: same data
files, same strict-QC masks, same cos(lat) area weights. Raw daily area-weighted
means are drawn thin, a 7-day centred running mean thick, so the day-to-day
variability and the seasonal trend are both visible.

Run on the server:
  python make_daily_curves.py OUT_PNG
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

OBS = ("/data/yangjinhui/surf_pytorch/surf_paper/observations/"
       "modis_cmg/modis_cmg_2018_cmfd_columns.nc")
PRIOR = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
         "modis_validation_1y_2018_spunup_20260822/surf_modis_observables.nc")
SLAB = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
        "slab_observables_final_20260824/surf_modis_observables_slab.nc")


def read_observables(path):
    with Dataset(path) as data:
        return {name: np.asarray(data[name][:], dtype=np.float64)
                for name in ("lst_day", "lst_night")}


def daily_mean(values, weights, valid):
    out = np.full(values.shape[0], np.nan)
    for day in range(values.shape[0]):
        sel = valid[day]
        if not sel.any():
            continue
        ww = weights[sel]
        out[day] = float((values[day][sel] * ww).sum() / ww.sum())
    return out


def running_mean(x, k=7):
    filled = np.nan_to_num(x, nan=np.nanmean(x))
    padded = np.pad(filled, k // 2, mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def main():
    out_png = Path(sys.argv[1])
    with Dataset(OBS) as observations:
        obs = {name: np.asarray(observations[name][:], dtype=np.float64)
               for name in ("lst_day", "lst_night")}
        masks = {name: np.asarray(observations[f"{name}_strict"][:]) == 1
                 for name in ("lst_day", "lst_night")}
        lat = np.asarray(observations["lat"][:], dtype=np.float64)
    weights = np.cos(np.deg2rad(lat))

    prior = read_observables(PRIOR)
    slab = read_observables(SLAB)

    import datetime as dt
    dates = [dt.date(2018, 1, 1) + dt.timedelta(days=i)
             for i in range(obs["lst_day"].shape[0])]

    series = {}
    for name in ("lst_day", "lst_night"):
        valid = (masks[name] & np.isfinite(obs[name])
                 & np.isfinite(prior[name]) & np.isfinite(slab[name]))
        series[name] = {
            "MODIS": daily_mean(obs[name], weights, valid),
            "SURF": daily_mean(prior[name], weights, valid),
            "SURF-Opt": daily_mean(slab[name], weights, valid),
        }

    styles = {"MODIS": ("k", 0.9), "SURF": ("0.55", 0.8), "SURF-Opt": ("tab:blue", 0.8)}
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 6.4), sharex=True,
                                constrained_layout=True)
    for axis, name, title in zip(axes, ("lst_day", "lst_night"),
                                 ("day pass", "night pass")):
        for label, (color, alpha) in styles.items():
            y = series[name][label]
            axis.plot(dates, y, color=color, alpha=0.25, linewidth=0.6)
            axis.plot(dates, running_mean(y), color=color, linewidth=1.6,
                      label=f"{label} (7-day mean)")
        axis.set_ylabel(f"{title} LST (K)", fontsize=9)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
        axis.legend(fontsize=7, ncol=3, loc="lower right")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    figure.suptitle("Daily area-weighted LST: MODIS vs SURF vs SURF-Opt "
                    "(strict QC, continuous 2018 integrations; thin = daily, "
                    "thick = 7-day running mean)", fontsize=11)
    figure.savefig(out_png, dpi=260)
    print("wrote", out_png)
    for name in ("lst_day", "lst_night"):
        for label in styles:
            y = series[name][label]
            print(f"{name} {label}: daily std {np.nanstd(y - running_mean(y)):.2f} K, "
                  f"range {np.nanmin(y):.1f}..{np.nanmax(y):.1f} K")


if __name__ == "__main__":
    main()
