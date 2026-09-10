#!/usr/bin/env python
"""True-terminal Fortran--PyTorch comparison for the chained multi-year run.

The per-year chained Fortran ``o_gg.nc`` archives end at day 360
(NFRPOS=480); the day-365 terminal state lives only in each year's
``restartout.nc``, whose variables lack a ``time`` dimension.  This script
wraps the terminal restart into a pseudo o_gg layout (time,lat,lon[,nlevs])
so that the field accessors, metrics, and map plotting of
``compare_autonomous_annual`` apply unchanged, and compares it with the
exact terminal Torch record of the continuous multi-year integration.

Environment:
  SURF_TORCH_MULTIYEAR   torch_offline_daily.npz of the multiyear run
  SURF_FORTRAN_YEARS     directory holding years/YYYY/restartout.nc
  SURF_MULTIYEAR_DATA    per-year dataset root (ntime metadata)
  SURF_COMPARE_OUTPUT    output directory
  SURF_YEARS             START-END (default 2009:2018)
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (needed by plot_maps import)
import numpy as np
from netCDF4 import Dataset

from compare_autonomous_annual import (
    FIELDS,
    SNOW_FIELDS,
    SOIL_T_FIELDS,
    SOIL_W_FIELDS,
    field_rows,
    fortran_at,
    plot_maps,
    torch_at,
)


def build_pseudo(restart_path: Path, output_path: Path) -> None:
    """Copy restart fields into a (time,lat,lon[,nlevs]) pseudo-o_gg file."""
    needed = sorted({field.fortran_name for field in FIELDS})
    with Dataset(restart_path) as src:
        src.set_auto_mask(False)
        if output_path.exists():
            output_path.unlink()
        with Dataset(output_path, "w") as dst:
            dst.createDimension("time", 1)
            dst.createDimension("lat", 1)
            ncol = None
            for name in needed:
                var = src.variables[name]
                if "lon" in var.dimensions:
                    ncol = var.shape[var.dimensions.index("lon")]
                    break
            if ncol is None:
                raise ValueError(f"no lon dimension found in {restart_path}")
            dst.createDimension("lon", ncol)
            if "nlevs" in src.dimensions:
                dst.createDimension("nlevs", len(src.dimensions["nlevs"]))
            for name in needed:
                var = src.variables[name]
                # restart layout: (nlevs?, lat, lon) -> pseudo: (time, lat, nlevs?, lon)
                order = ["lat"] + [d for d in var.dimensions if d not in ("lat", "lon")] + ["lon"]
                data = np.transpose(np.asarray(var[...], dtype=np.float64),
                                    [var.dimensions.index(d) for d in order])[np.newaxis]
                dims = ("time", "lat") + tuple(
                    d for d in order if d not in ("lat", "lon")) + ("lon",)
                out = dst.createVariable(name, "f8", dims)
                out[0] = data


def main() -> None:
    torch_path = Path(os.environ["SURF_TORCH_MULTIYEAR"])
    years_root = Path(os.environ["SURF_FORTRAN_YEARS"])
    data_root = Path(os.environ["SURF_MULTIYEAR_DATA"])
    output_dir = Path(os.environ["SURF_COMPARE_OUTPUT"])
    years = [int(v) for v in os.environ.get("SURF_YEARS", "2009:2018").replace(":", "-").split("-")]
    years = list(range(years[0], years[1] + 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(np.load(data_root / f"cmfd_valid_columns_{years[0]}_dataset/lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(data_root / f"cmfd_valid_columns_{years[0]}_dataset/lon.npy", mmap_mode="r"), dtype=np.float64)

    offset = 0
    offsets: dict[int, int] = {}
    for year in years:
        metadata = json.loads((data_root / f"cmfd_valid_columns_{year}_dataset/metadata.json").read_text())
        offset += int(metadata["ntime"]) * 6
        offsets[year] = offset
    last_year = years[-1]
    terminal_step = offsets[last_year]  # cumulative steps through the end of the last year

    rows: list[dict] = []
    swe_rows: list[dict] = []
    with np.load(torch_path) as torch_data:
        saved = {int(step): index for index, step in enumerate(torch_data["step"])}
        # Per-year true-terminal (day-365) SWE comparison from the restart chain.
        for year in years:
            restart = years_root / str(year) / "restartout.nc"
            if not restart.exists():
                print(f"missing {restart}, skipped")
                continue
            year_step = offsets[year]
            if year_step not in saved:
                print(f"no Torch record at step {year_step} (year {year}), skipped")
                continue
            pseudo = output_dir / f"terminal_restart_{year}_pseudo.nc"
            build_pseudo(restart, pseudo)
            with Dataset(pseudo) as dataset:
                dataset.set_auto_mask(False)
                torch_record = saved[year_step]
                day = year_step * 1800.0 / 86_400.0
                rows.extend(field_rows(torch_data, dataset, torch_record, 0, year_step, day, lat, lon))
                if year == last_year:
                    plot_maps(
                        torch_data, dataset, torch_record, 0,
                        SOIL_T_FIELDS[:2] + SOIL_W_FIELDS[:2] + SNOW_FIELDS[:2], lat, lon,
                        output_dir / "multiyear_true_terminal_maps.png",
                        f"True terminal state, 31 December {last_year} (day {day:.0f})",
                    )
                swe_t = torch_at(torch_data, SNOW_FIELDS[0], saved[year_step])
                swe_f = fortran_at(dataset, SNOW_FIELDS[0], 0)
                swe_rows.append({
                    "year": year,
                    "fortran_mean_swe": float(np.mean(swe_f)),
                    "torch_mean_swe": float(np.mean(swe_t)),
                    "fortran_max_swe": float(np.max(swe_f)),
                    "torch_max_swe": float(np.max(swe_t)),
                    "rmse": float(np.sqrt(np.mean((swe_t - swe_f) ** 2))),
                    "max_abs_diff": float(np.max(np.abs(swe_t - swe_f))),
                })

    json.dump(rows, (output_dir / "multiyear_terminal_metrics.json").open("w"), indent=1)
    with (output_dir / "multiyear_terminal_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "multiyear_terminal_swe.json").write_text(json.dumps(swe_rows, indent=2) + "\n")

    # Annual-terminal error evolution across the decade (exact day-365 states).
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, constrained_layout=True)
    total_days = offsets[last_year] * 1800.0 / 86_400.0
    groups = ((SOIL_T_FIELDS + SOIL_W_FIELDS, "Soil states"), (SNOW_FIELDS, "Snow states"))
    colors = plt.get_cmap("tab10").colors
    for row_index, (fields, title) in enumerate(groups):
        for column_index, (metric_name, ylabel) in enumerate((
            ("rmse", "RMSE (native unit)"), ("relative_rmse_percent", "Relative RMSE (%)"),
        )):
            axis = axes[row_index, column_index]
            for color, field in zip(colors, fields):
                domain = "snow_fraction_gt_1e-3" if field.label in {"snow_t", "snow_rho", "snow_alb"} else "all_columns"
                points = [p for p in rows if p["field"] == field.label and p["domain"] == domain]
                axis.semilogy([float(p["day"]) for p in points], [float(p[metric_name]) for p in points],
                              color=color, marker="o", markersize=3, linewidth=1.0, label=field.title)
            axis.set(title=f"{title}: {ylabel}", ylabel=ylabel, xlim=(0.0, total_days))
            axis.grid(alpha=0.25, which="both")
            axis.legend(fontsize=6.5, ncol=2, frameon=False)
    for axis in axes[1]:
        axis.set_xlabel("Continuous integration day (annual terminal states)")
        axis.set_xticks([offsets[years[0] + k] * 1800.0 / 86_400.0 for k in range(len(years))])
        axis.set_xticklabels([str(years[0] + k) for k in range(len(years))], fontsize=7)
    figure.suptitle(f"Annual-terminal (31 December) Fortran-PyTorch errors, {years[0]}-{years[-1]}", fontsize=13)
    figure.savefig(output_dir / "multiyear_annual_terminal_error_growth.png", dpi=260)
    plt.close(figure)
    print(json.dumps({"terminal_step": terminal_step, "rows": len(rows), "swe_years": len(swe_rows)}, indent=1))


if __name__ == "__main__":
    main()
