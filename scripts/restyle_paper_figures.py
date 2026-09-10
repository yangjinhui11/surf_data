#!/usr/bin/env python3
"""Re-author the core manuscript figures at print scale.

The GMD (copernicus, manuscript) class places ``width=\\linewidth`` figures at
6.97 in.  The original generators author figures 11-15 in wide, so every font
shrinks by a factor 0.27-0.63 in print.  This script redraws the six core
figures at exactly the printed size, so the fonts below are the effective
printed sizes, and regenerates them from saved metrics/terminal states without
repeating any Fortran or PyTorch integration.

Outputs (same filenames as the originals, written next to them):
  equiv_controlled_full_30d_current_20260807/controlled_30d_error_growth.png
  equiv_controlled_full_30d_current_20260807/controlled_30d_terminal_maps.png
  repro_autonomous_1y_rpsfr2_20260807/comparison/annual_autonomous_error_growth.png
  repro_autonomous_1y_rpsfr2_20260807/comparison/annual_autonomous_terminal_{soil_temperature,soil_water,snow}_maps.png
  autonomous_multiyear_2009_2018_compare/multiyear_annual_terminal_error_growth.png
  cmfd_offline_runner/surf_cpu_gpu_scaling_current.png
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from netCDF4 import Dataset

PRINT_WIDTH = 6.97  # inches; copernicus manuscript \linewidth

CONTROLLED_DIR = Path("/data/yangjinhui/surf_pytorch/surf_paper/results/equiv_controlled_full_30d_current_20260807")
CONTROLLED_FORTRAN = Path("/data/yangjinhui/surf_pytorch/full_domain_30d_rebuilt_capture_20260806/o_gg.nc")
ANNUAL_DIR = Path("/data/yangjinhui/surf_pytorch/surf_paper/experiments/repro_autonomous_1y_rpsfr2_20260807")
ANNUAL_COMPARISON = ANNUAL_DIR / "comparison"
ANNUAL_FINAL_FORTRAN = Path("/data/yangjinhui/surf_pytorch/surf_paper/experiments/fortran_reference_1y_2018/terminal_day365/o_gg.nc")
MULTIYEAR_DIR = Path("/data/yangjinhui/surf_pytorch/surf_paper/experiments/autonomous_multiyear_2009_2018_compare")
PERF_JSON = Path("/data/yangjinhui/surf_pytorch/cmfd_offline_runner/performance_results_clean.json")
DATA_DIR = Path("/data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset")


def load_module(name: str, path: Path):
    import sys
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANNUAL_MOD = load_module("annual_mod", Path("/data/yangjinhui/surf_pytorch/cmfd_offline_runner/compare_autonomous_annual.py"))
CONTROLLED_MOD = load_module(
    "controlled_mod",
    CONTROLLED_DIR / "scripts" / "compare_full_domain_controlled.py",
)


def print_style() -> None:
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "figure.dpi": 100,
    })


LAYER_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
FIELD_STYLE: dict[str, dict[str, str]] = {
    "avg_skin_t": {"label": "Surface temp.", "color": "#000000", "ls": "-", "marker": "none", "domain": "all_columns"},
    "snow_w": {"label": "SWE", "color": "#0072B2", "ls": "-", "marker": "o", "domain": "all_columns"},
    "snow_t": {"label": "Snow temp.", "color": "#D55E00", "ls": "-", "marker": "s", "domain": "snow_fraction_gt_1e-3"},
    "snow_rho": {"label": "Snow density", "color": "#009E73", "ls": "-", "marker": "^", "domain": "snow_fraction_gt_1e-3"},
    "snow_alb": {"label": "Snow albedo", "color": "#CC79A7", "ls": "-", "marker": "D", "domain": "snow_fraction_gt_1e-3"},
}
for layer, color in enumerate(LAYER_COLORS):
    FIELD_STYLE[f"soil_t_l{layer + 1}"] = {"label": f"Soil temp. L{layer + 1}", "color": color, "ls": "-", "marker": "none", "domain": "all_columns"}
    FIELD_STYLE[f"soil_w_l{layer + 1}"] = {"label": f"Soil water L{layer + 1}", "color": color, "ls": "--", "marker": "none", "domain": "all_columns"}


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def series(rows: list[dict], field: str, metric: str) -> tuple[list[float], list[float]]:
    style = FIELD_STYLE[field]
    points = [row for row in rows if row["field"] == field and row["domain"] == style["domain"]]
    points.sort(key=lambda row: float(row["day"]))
    return [float(row["day"]) for row in points], [float(row[metric]) for row in points]


def draw_curves(axis, rows: list[dict], fields: list[str], metric: str, markersize: float = 3.0) -> None:
    for field in fields:
        style = FIELD_STYLE[field]
        x, y = series(rows, field, metric)
        axis.semilogy(
            x, y, color=style["color"], linestyle=style["ls"], linewidth=1.1,
            marker=style["marker"], markersize=markersize, markeredgewidth=0.0,
            label=style["label"],
        )


def add_legend_with_headroom(axis, columns: int) -> None:
    ymin, ymax = axis.get_ylim()
    axis.set_ylim(ymin, ymax * (10.0 if columns == 2 else 6.0))
    axis.legend(
        ncol=columns, frameon=False, loc="upper left", handlelength=1.9,
        columnspacing=1.0, handletextpad=0.5, borderaxespad=0.2, labelspacing=0.35,
    )


# ---------------------------------------------------------------- growth figures

def plot_controlled_growth() -> None:
    rows = read_csv_rows(CONTROLLED_DIR / "controlled_30d_metrics.csv")
    fields = ["avg_skin_t", "soil_t_l1", "soil_w_l1", "snow_w", "snow_t", "snow_rho", "snow_alb"]
    figure, axes = plt.subplots(2, 1, figsize=(PRINT_WIDTH, 4.3), sharex=True, constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "rmse", "Absolute RMSE"),
        (axes[1], "relative_rmse_percent", "Relative RMSE"),
    ):
        draw_curves(axis, rows, fields, metric, markersize=2.6)
        axis.set(ylabel="RMSE (native unit)" if metric == "rmse" else "Relative RMSE (%)", xlim=(0.5, 30.5))
        axis.set_title(title, pad=3)
        axis.grid(alpha=0.3, which="both", linewidth=0.5)
        add_legend_with_headroom(axis, columns=2)
    axes[1].set_xlabel("Integration day")
    figure.savefig(CONTROLLED_DIR / "controlled_30d_error_growth.png", dpi=300)
    plt.close(figure)
    print("controlled_30d_error_growth.png", len(rows), "rows")


def plot_annual_growth() -> None:
    rows = read_csv_rows(ANNUAL_COMPARISON / "annual_autonomous_metrics.csv")
    soil = [f"soil_t_l{layer + 1}" for layer in range(4)] + [f"soil_w_l{layer + 1}" for layer in range(4)]
    interleaved = [field for pair in zip(soil[:4], soil[4:]) for field in pair]
    snow = ["snow_w", "snow_t", "snow_rho", "snow_alb"]
    figure, axes = plt.subplots(2, 2, figsize=(PRINT_WIDTH, 4.7), sharex=True, constrained_layout=True)
    for row_index, (group, fields) in enumerate((("Soil", interleaved), ("Snow", snow))):
        for column_index, (metric, metric_title) in enumerate((
            ("rmse", "absolute RMSE"), ("relative_rmse_percent", "relative RMSE"),
        )):
            axis = axes[row_index, column_index]
            draw_curves(axis, rows, fields, metric, markersize=2.4)
            axis.set_title(f"{group} states: {metric_title}", pad=3)
            axis.set(ylabel="RMSE (native unit)" if metric == "rmse" else "Relative RMSE (%)", xlim=(0.0, 370.0))
            axis.grid(alpha=0.3, which="both", linewidth=0.5)
            add_legend_with_headroom(axis, columns=2 if group == "Soil" else 1)
    for axis in axes[1]:
        axis.set_xlabel("Integration day")
    figure.savefig(ANNUAL_COMPARISON / "annual_autonomous_error_growth.png", dpi=300)
    plt.close(figure)
    print("annual_autonomous_error_growth.png", len(rows), "rows")


def plot_multiyear_growth() -> None:
    rows = json.loads((MULTIYEAR_DIR / "multiyear_terminal_metrics.json").read_text())
    soil = [f"soil_t_l{layer + 1}" for layer in range(4)] + [f"soil_w_l{layer + 1}" for layer in range(4)]
    interleaved = [field for pair in zip(soil[:4], soil[4:]) for field in pair]
    snow = ["snow_w", "snow_t", "snow_rho", "snow_alb"]
    days = sorted({float(row["day"]) for row in rows})
    years = [2009 + index for index in range(len(days))]
    if len(days) != 10:
        raise ValueError(f"expected ten annual terminal states, found {len(days)}")
    figure, axes = plt.subplots(2, 2, figsize=(PRINT_WIDTH, 4.7), constrained_layout=True)
    for row_index, (group, fields) in enumerate((("Soil", interleaved), ("Snow", snow))):
        for column_index, (metric, metric_title) in enumerate((
            ("rmse", "absolute RMSE"), ("relative_rmse_percent", "relative RMSE"),
        )):
            axis = axes[row_index, column_index]
            draw_curves(axis, rows, fields, metric, markersize=3.4)
            axis.set_title(f"{group} states: {metric_title}", pad=3)
            axis.set(ylabel="RMSE (native unit)" if metric == "rmse" else "Relative RMSE (%)")
            axis.set_xticks(days)
            axis.set_xticklabels([str(year) for year in years])
            axis.set_xlim(days[0] - 260.0, days[-1] + 260.0)
            axis.grid(alpha=0.3, which="both", linewidth=0.5)
            add_legend_with_headroom(axis, columns=2 if group == "Soil" else 1)
    for axis in axes[1]:
        axis.set_xlabel("Year (31 December terminal states)")
    figure.savefig(MULTIYEAR_DIR / "multiyear_annual_terminal_error_growth.png", dpi=300)
    plt.close(figure)
    print("multiyear_annual_terminal_error_growth.png", len(rows), "rows,", len(days), "years")


# -------------------------------------------------------------------- map figures

def plot_maps_print(
    module,
    torch_npz: np.lib.npyio.NpzFile,
    reference: Dataset,
    torch_record: int,
    fortran_record: int,
    fields: tuple,
    lat: np.ndarray,
    lon: np.ndarray,
    output: Path,
    height: float,
) -> None:
    swe_field = module.SNOW_FIELDS[0] if hasattr(module, "SNOW_FIELDS") else module.FIELDS[9]
    rho_field = module.SNOW_FIELDS[2] if hasattr(module, "SNOW_FIELDS") else module.FIELDS[11]
    swe_candidate = module.torch_at(torch_npz, swe_field, torch_record)
    swe_truth = module.fortran_at(reference, swe_field, fortran_record)
    rho_candidate = module.torch_at(torch_npz, rho_field, torch_record)
    rho_truth = module.fortran_at(reference, rho_field, fortran_record)
    fraction_candidate = module.snow_fraction(swe_candidate, rho_candidate)
    fraction_truth = module.snow_fraction(swe_truth, rho_truth)
    figure, axes = plt.subplots(len(fields), 3, figsize=(PRINT_WIDTH, height), constrained_layout=True)
    for row_index, (row_axes, field) in enumerate(zip(np.atleast_2d(axes), fields)):
        candidate = module.torch_at(torch_npz, field, torch_record)
        truth = module.fortran_at(reference, field, fortran_record)
        domain = "snow_fraction_gt_1e-3" if field.snow_related and field.torch_name != "snow_w" else "all_columns"
        mask = module.valid_mask(
            candidate, truth, swe_candidate, swe_truth, fraction_candidate, fraction_truth, domain,
        )
        candidate = np.where(mask, candidate, np.nan)
        truth = np.where(mask, truth, np.nan)
        difference = candidate - truth
        state_values = np.concatenate((candidate[np.isfinite(candidate)], truth[np.isfinite(truth)]))
        lo, hi = np.quantile(state_values, (0.01, 0.99))
        limit = max(float(np.nanquantile(np.abs(difference), 0.995)), np.finfo(np.float64).eps)
        panels = (
            (truth, field.title, "viridis", None, lo, hi),
            (candidate, "PyTorch", "viridis", None, lo, hi),
            (difference, "PyTorch \u2212 Fortran", "RdBu_r", TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), None, None),
        )
        for column_index, (axis, (values, title, cmap, norm, vmin, vmax)) in enumerate(zip(row_axes, panels)):
            x, y, image = module.grid(values, lat, lon)
            if norm is None:
                mesh = axis.pcolormesh(x, y, image, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                mesh = axis.pcolormesh(x, y, image, shading="nearest", cmap=cmap, norm=norm)
            axis.set_title(title, fontsize=7.0, pad=2.5)
            axis.set_aspect("equal")
            axis.tick_params(labelsize=6.0)
            axis.xaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_locator(MaxNLocator(4))
            if column_index == 0:
                axis.set_ylabel("Latitude (deg N)", fontsize=6.5)
            if row_index == len(fields) - 1:
                axis.set_xlabel("Longitude (deg E)", fontsize=6.5)
            colorbar = figure.colorbar(mesh, ax=axis, shrink=0.9, pad=0.02)
            colorbar.set_label(field.unit, fontsize=6.0)
            colorbar.ax.tick_params(labelsize=5.5)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    print(output.name, len(fields), "field rows")


def plot_controlled_maps(lat: np.ndarray, lon: np.ndarray) -> None:
    fields = (
        CONTROLLED_MOD.FIELDS[0], CONTROLLED_MOD.FIELDS[1], CONTROLLED_MOD.FIELDS[5],
        CONTROLLED_MOD.FIELDS[9], CONTROLLED_MOD.FIELDS[10], CONTROLLED_MOD.FIELDS[11],
        CONTROLLED_MOD.FIELDS[12],
    )
    with np.load(CONTROLLED_DIR / "torch_daily_controlled.npz") as torch_npz, Dataset(CONTROLLED_FORTRAN) as reference:
        torch_steps = np.asarray(torch_npz["step"], dtype=int)
        fortran_steps = np.asarray(reference.variables["timestp"][:], dtype=int)
        plot_maps_print(
            CONTROLLED_MOD, torch_npz, reference, int(torch_steps.size - 1), int(fortran_steps.size - 1),
            fields, lat, lon, CONTROLLED_DIR / "controlled_30d_terminal_maps.png", height=6.8,
        )


def plot_annual_maps(lat: np.ndarray, lon: np.ndarray) -> None:
    summary = json.loads((ANNUAL_COMPARISON / "annual_autonomous_metrics.json").read_text())
    terminal_step = int(summary["terminal_step"])
    groups = (
        (ANNUAL_MOD.SOIL_T_FIELDS, "annual_autonomous_terminal_soil_temperature_maps.png", 4.6),
        (ANNUAL_MOD.SOIL_W_FIELDS, "annual_autonomous_terminal_soil_water_maps.png", 4.6),
        (ANNUAL_MOD.SNOW_FIELDS, "annual_autonomous_terminal_snow_maps.png", 4.6),
    )
    with np.load(ANNUAL_DIR / "torch_offline_daily.npz") as torch_npz, Dataset(ANNUAL_FINAL_FORTRAN) as reference:
        torch_steps = np.asarray(torch_npz["step"], dtype=int)
        matches = np.flatnonzero(torch_steps == terminal_step)
        if matches.size != 1:
            raise ValueError(f"expected one Torch record at step {terminal_step}, found {matches.size}")
        torch_record = int(matches[0])
        fortran_steps = np.asarray(reference.variables["timestp"][:], dtype=int)
        fortran_record = int(np.argmax(fortran_steps))
        for fields, name, height in groups:
            plot_maps_print(
                ANNUAL_MOD, torch_npz, reference, torch_record, fortran_record,
                fields, lat, lon, ANNUAL_COMPARISON / name, height=height,
            )


# ----------------------------------------------------------- performance figure

def plot_performance() -> None:
    report = json.loads(PERF_JSON.read_text())
    steps = int(report["configuration"]["steps"])
    columns = np.asarray(report["configuration"]["columns"], dtype=int)
    records = {}
    for row in report["median_results"]:
        records[(str(row["mode"]), int(row["columns"]))] = row

    def throughput(mode: str):
        median = np.asarray([float(records[(mode, int(n))]["integration_seconds_median"]) for n in columns])
        minimum = np.asarray([float(records[(mode, int(n))]["integration_seconds_min"]) for n in columns])
        maximum = np.asarray([float(records[(mode, int(n))]["integration_seconds_max"]) for n in columns])
        rate = columns * steps / median
        return rate, np.vstack((rate - columns * steps / maximum, columns * steps / minimum - rate))

    cpu_rate, cpu_error = throughput("cpu")
    gpu_rate, gpu_error = throughput("gpu")
    speedup = np.asarray([
        float(records[("cpu", int(n))]["integration_seconds_median"]) / float(records[("gpu", int(n))]["integration_seconds_median"])
        for n in columns
    ])
    figure, (ax_rate, ax_speedup) = plt.subplots(1, 2, figsize=(PRINT_WIDTH, 2.95), constrained_layout=True)
    ax_rate.errorbar(columns, cpu_rate, yerr=cpu_error, fmt="o-", color="#ba5b30", lw=1.4, ms=3.5, capsize=2.5, label="CPU (32 threads)")
    ax_rate.errorbar(columns, gpu_rate, yerr=gpu_error, fmt="s-", color="#1976a2", lw=1.4, ms=3.5, capsize=2.5, label="A100 GPU")
    ax_rate.set_xscale("log")
    ax_rate.set_yscale("log")
    ax_rate.set_xlabel("Active land columns")
    ax_rate.set_ylabel("Throughput (column-steps s$^{-1}$)")
    ax_rate.set_title(f"{steps}-step integration throughput", pad=3)
    ax_rate.grid(True, which="both", alpha=0.3, lw=0.5)
    ax_rate.legend(frameon=False, loc="upper left", fontsize=6.5, handlelength=1.6)
    ax_speedup.plot(columns, speedup, "o-", color="#147a5b", lw=1.4, ms=3.5)
    ax_speedup.axhline(1.0, color="0.35", lw=0.8, ls="--")
    ax_speedup.set_xscale("log")
    ax_speedup.set_xlabel("Active land columns")
    ax_speedup.set_ylabel("GPU / CPU speed-up")
    ax_speedup.set_title("GPU advantage vs. batch size", pad=3)
    ax_speedup.set_ylim(0.0, float(np.max(speedup)) * 1.2)
    ax_speedup.grid(True, which="both", alpha=0.3, lw=0.5)
    for x, y in zip(columns, speedup):
        ax_speedup.annotate(f"{y:.1f}\u00d7", (x, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.5)
    figure.savefig(Path("/data/yangjinhui/surf_pytorch/cmfd_offline_runner/surf_cpu_gpu_scaling_current.png"), dpi=300)
    plt.close(figure)
    print("surf_cpu_gpu_scaling_current.png", list(speedup.round(2)))


def main() -> None:
    print_style()
    lat = np.asarray(np.load(DATA_DIR / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(DATA_DIR / "lon.npy", mmap_mode="r"), dtype=np.float64)
    plot_controlled_growth()
    plot_annual_growth()
    plot_multiyear_growth()
    plot_controlled_maps(lat, lon)
    plot_annual_maps(lat, lon)
    plot_performance()


if __name__ == "__main__":
    main()
