#!/usr/bin/env python3
"""Compare a captured-boundary PyTorch SURF replay with OSM/Fortran output.

The comparison is deliberately independent of the experiment runner.  It
matches snapshots through the explicit ``timestp`` coordinate written by OSM,
reports both all-column and snow-active metrics, and records the location of
the largest absolute discrepancy for every field and output time.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from netCDF4 import Dataset


@dataclass(frozen=True)
class Field:
    torch_name: str
    fortran_name: str
    title: str
    unit: str
    level: int | None = None
    snow_related: bool = False

    @property
    def label(self) -> str:
        return self.torch_name if self.level is None else f"{self.torch_name}_l{self.level + 1}"


FIELDS = (
    Field("avg_skin_t", "AvgSurfT", "Surface temperature", "K"),
    *(Field("soil_t", "SoilTemp", f"Soil temperature, layer {level + 1}", "K", level) for level in range(4)),
    *(Field("soil_w", "SoilMoist", f"Soil water, layer {level + 1}", "kg m-2", level) for level in range(4)),
    Field("snow_w", "SWE", "Snow water equivalent", "kg m-2", snow_related=True),
    Field("snow_t", "SnowT", "Snow temperature", "K", snow_related=True),
    Field("snow_rho", "snowdens", "Snow density", "kg m-3", snow_related=True),
    Field("snow_alb", "SAlbedo", "Snow albedo", "1", snow_related=True),
)


def load_environment() -> tuple[Path, Path, Path, Path]:
    try:
        torch_file = Path(os.environ["SURF_TORCH_CONTROLLED"])
        fortran_file = Path(os.environ["SURF_FORTRAN_CONTROLLED"])
        data_dir = Path(os.environ["SURF_DATA"])
        output_dir = Path(os.environ["SURF_COMPARE_OUTPUT"])
    except KeyError as exc:
        raise SystemExit(f"Missing required environment variable: {exc.args[0]}") from exc
    return torch_file, fortran_file, data_dir, output_dir


def torch_at(source: np.lib.npyio.NpzFile, field: Field, record: int) -> np.ndarray:
    values = np.asarray(source[field.torch_name][record], dtype=np.float64)
    return values[:, field.level] if field.level is not None else values.reshape(-1)


def fortran_at(source: Dataset, field: Field, record: int) -> np.ndarray:
    variable = source.variables[field.fortran_name]
    selector: list[object] = [slice(None)] * variable.ndim
    selector[variable.dimensions.index("time")] = record
    selector[variable.dimensions.index("lat")] = 0
    if field.level is not None:
        selector[variable.dimensions.index("nlevs")] = field.level
    return np.asarray(variable[tuple(selector)], dtype=np.float64).reshape(-1)


def snow_fraction(swe: np.ndarray, density: np.ndarray) -> np.ndarray:
    return np.clip(10.0 * swe / np.maximum(density, np.finfo(np.float64).tiny), 0.0, 1.0)


def valid_mask(
    candidate: np.ndarray,
    truth: np.ndarray,
    swe_candidate: np.ndarray,
    swe_truth: np.ndarray,
    snow_fraction_candidate: np.ndarray,
    snow_fraction_truth: np.ndarray,
    domain: str,
) -> np.ndarray:
    mask = np.isfinite(candidate) & np.isfinite(truth)
    if domain == "snow_present":
        mask &= np.maximum(swe_candidate, swe_truth) > 1.0e-8
    elif domain == "snow_fraction_gt_1e-3":
        # Snow temperature and density become ill-conditioned as the snow
        # heat capacity vanishes.  This model-native area-fraction criterion
        # retains columns with at least 0.1% snow cover while the unfiltered
        # all-column metrics remain available for complete disclosure.
        mask &= np.maximum(snow_fraction_candidate, snow_fraction_truth) > 1.0e-3
    elif domain != "all_columns":
        raise ValueError(f"Unknown comparison domain: {domain}")
    return mask


def metric(
    candidate: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> dict[str, float | int]:
    if not np.any(mask):
        return {
            "n_columns": 0,
            "rmse": float("nan"),
            "relative_rmse_percent": float("nan"),
            "mean_abs_error": float("nan"),
            "max_abs_error": float("nan"),
            "max_index": -1,
            "max_lat": float("nan"),
            "max_lon": float("nan"),
            "max_torch": float("nan"),
            "max_fortran": float("nan"),
        }
    diff = candidate[mask] - truth[mask]
    source_index = np.flatnonzero(mask)
    max_local = int(np.argmax(np.abs(diff)))
    max_index = int(source_index[max_local])
    rmse = float(np.sqrt(np.mean(diff**2)))
    scale = float(np.sqrt(np.mean(truth[mask] ** 2)))
    return {
        "n_columns": int(diff.size),
        "rmse": rmse,
        "relative_rmse_percent": 100.0 * rmse / scale if scale > 0.0 else float("nan"),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.abs(diff[max_local])),
        "max_index": max_index,
        "max_lat": float(lat[max_index]),
        "max_lon": float(lon[max_index]),
        "max_torch": float(candidate[max_index]),
        "max_fortran": float(truth[max_index]),
    }


def field_rows(
    torch: np.lib.npyio.NpzFile,
    reference: Dataset,
    torch_record: int,
    fortran_record: int,
    step: int,
    day: float,
    lat: np.ndarray,
    lon: np.ndarray,
) -> list[dict[str, float | int | str]]:
    swe_candidate = torch_at(torch, FIELDS[9], torch_record)
    swe_truth = fortran_at(reference, FIELDS[9], fortran_record)
    rho_candidate = torch_at(torch, FIELDS[11], torch_record)
    rho_truth = fortran_at(reference, FIELDS[11], fortran_record)
    snow_fraction_candidate = snow_fraction(swe_candidate, rho_candidate)
    snow_fraction_truth = snow_fraction(swe_truth, rho_truth)
    rows: list[dict[str, float | int | str]] = []
    for field in FIELDS:
        candidate = torch_at(torch, field, torch_record)
        truth = fortran_at(reference, field, fortran_record)
        domains = ("all_columns", "snow_present", "snow_fraction_gt_1e-3") if field.snow_related else ("all_columns",)
        for domain in domains:
            row: dict[str, float | int | str] = {
                "step": step,
                "day": day,
                "field": field.label,
                "title": field.title,
                "unit": field.unit,
                "domain": domain,
            }
            row.update(metric(
                candidate,
                truth,
                valid_mask(
                    candidate,
                    truth,
                    swe_candidate,
                    swe_truth,
                    snow_fraction_candidate,
                    snow_fraction_truth,
                    domain,
                ),
                lat,
                lon,
            ))
            max_index = int(row["max_index"])
            row.update({
                "torch_swe_at_max": float(swe_candidate[max_index]) if max_index >= 0 else float("nan"),
                "fortran_swe_at_max": float(swe_truth[max_index]) if max_index >= 0 else float("nan"),
                "torch_snow_fraction_at_max": float(snow_fraction_candidate[max_index]) if max_index >= 0 else float("nan"),
                "fortran_snow_fraction_at_max": float(snow_fraction_truth[max_index]) if max_index >= 0 else float("nan"),
            })
            rows.append(row)
    return rows


def grid(values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.unique(lat)
    x = np.unique(lon)
    image = np.full((y.size, x.size), np.nan, dtype=np.float64)
    image[np.searchsorted(y, lat), np.searchsorted(x, lon)] = values
    return x, y, image


def plot_growth(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    selected = ("avg_skin_t", "soil_t_l1", "soil_w_l1", "snow_w", "snow_t", "snow_rho", "snow_alb")
    labels = {field.label: field.title for field in FIELDS}
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#000000")
    for axis, column, ylabel in (
        (axes[0], "rmse", "RMSE (native unit)"),
        (axes[1], "relative_rmse_percent", "Relative RMSE (%)"),
    ):
        for color, field_name in zip(colors, selected):
            domain = "snow_fraction_gt_1e-3" if field_name in ("snow_t", "snow_rho", "snow_alb") else "all_columns"
            points = [row for row in rows if row["field"] == field_name and row["domain"] == domain]
            axis.semilogy(
                [float(point["day"]) for point in points],
                [float(point[column]) for point in points],
                color=color,
                linewidth=1.25,
                label=labels[field_name],
            )
        axis.set(ylabel=ylabel, xlim=(1.0, 30.0))
        axis.grid(alpha=0.25, which="both")
        axis.legend(ncol=2, fontsize=8, frameon=False)
    axes[1].set_xlabel("Integration day")
    figure.suptitle("Controlled full-domain SURF replay: Fortran-PyTorch error evolution", fontsize=13)
    figure.savefig(output_dir / "controlled_30d_error_growth.png", dpi=260)
    plt.close(figure)


def plot_terminal_maps(
    torch: np.lib.npyio.NpzFile,
    reference: Dataset,
    torch_record: int,
    fortran_record: int,
    lat: np.ndarray,
    lon: np.ndarray,
    output_dir: Path,
) -> None:
    fields = (FIELDS[0], FIELDS[1], FIELDS[5], FIELDS[9], FIELDS[10], FIELDS[11], FIELDS[12])
    figure, axes = plt.subplots(len(fields), 3, figsize=(15, 25), constrained_layout=True)
    for row_axes, field in zip(axes, fields):
        candidate = torch_at(torch, field, torch_record)
        truth = fortran_at(reference, field, fortran_record)
        swe_candidate = torch_at(torch, FIELDS[9], torch_record)
        swe_truth = fortran_at(reference, FIELDS[9], fortran_record)
        rho_candidate = torch_at(torch, FIELDS[11], torch_record)
        rho_truth = fortran_at(reference, FIELDS[11], fortran_record)
        domain = "snow_fraction_gt_1e-3" if field.snow_related and field.torch_name != "snow_w" else "all_columns"
        mask = valid_mask(
            candidate,
            truth,
            swe_candidate,
            swe_truth,
            snow_fraction(swe_candidate, rho_candidate),
            snow_fraction(swe_truth, rho_truth),
            domain,
        )
        candidate = np.where(mask, candidate, np.nan)
        truth = np.where(mask, truth, np.nan)
        diff = candidate - truth
        values = np.concatenate((candidate[np.isfinite(candidate)], truth[np.isfinite(truth)]))
        lo, hi = np.quantile(values, (0.01, 0.99))
        error_limit = max(float(np.nanquantile(np.abs(diff), 0.995)), np.finfo(np.float64).eps)
        panels = (
            (truth, "Fortran", "viridis", None, lo, hi),
            (candidate, "PyTorch", "viridis", None, lo, hi),
            (diff, "PyTorch - Fortran", "RdBu_r", TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit), None, None),
        )
        for axis, (values, title, cmap, norm, vmin, vmax) in zip(row_axes, panels):
            x, y, image = grid(values, lat, lon)
            mesh = axis.pcolormesh(x, y, image, shading="nearest", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
            axis.set(title=f"{field.title}: {title}", xlabel="Longitude (deg E)", ylabel="Latitude (deg N)", aspect="equal")
            figure.colorbar(mesh, ax=axis, shrink=0.82, label=field.unit)
    figure.suptitle("Day 30 controlled full-domain SURF comparison", fontsize=14)
    figure.savefig(output_dir / "controlled_30d_terminal_maps.png", dpi=220)
    plt.close(figure)


def main() -> None:
    torch_path, fortran_path, data_dir, output_dir = load_environment()
    output_dir.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(np.load(data_dir / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(data_dir / "lon.npy", mmap_mode="r"), dtype=np.float64)
    with np.load(torch_path) as torch, Dataset(fortran_path) as reference:
        torch_steps = np.asarray(torch["step"], dtype=int)
        fortran_steps = np.asarray(reference.variables["timestp"][:], dtype=int)
        if lat.size != torch["avg_skin_t"].shape[1] or lon.size != lat.size:
            raise ValueError("Latitude/longitude arrays do not match the Torch column count")
        rows: list[dict[str, float | int | str]] = []
        matches: list[tuple[int, int, int]] = []
        for torch_record, step in enumerate(torch_steps):
            indices = np.flatnonzero(fortran_steps == step)
            if indices.size != 1:
                raise ValueError(f"Expected exactly one Fortran output at step {step}, found {indices.size}")
            fortran_record = int(indices[0])
            matches.append((torch_record, fortran_record, int(step)))
            day = float(reference.variables["time"][fortran_record]) / 86400.0
            rows.extend(field_rows(torch, reference, torch_record, fortran_record, int(step), day, lat, lon))
        if not matches:
            raise ValueError("The Torch archive contains no snapshots")
        fieldnames = list(rows[0])
        with (output_dir / "controlled_30d_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        terminal_step = matches[-1][2]
        terminal = [row for row in rows if int(row["step"]) == terminal_step]
        summary = {
            "torch_file": str(torch_path),
            "fortran_file": str(fortran_path),
            "n_columns": int(lat.size),
            "snapshot_steps": [int(item[2]) for item in matches],
            "terminal_step": terminal_step,
            "terminal_day": float(reference.variables["time"][matches[-1][1]]) / 86400.0,
            "terminal_metrics": terminal,
        }
        (output_dir / "controlled_30d_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
        plot_growth(rows, output_dir)
        plot_terminal_maps(torch, reference, matches[-1][0], matches[-1][1], lat, lon, output_dir)
    print(json.dumps({"n_columns": int(lat.size), "terminal_step": terminal_step, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
