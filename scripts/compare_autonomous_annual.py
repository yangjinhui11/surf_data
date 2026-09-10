#!/usr/bin/env python3
"""Compare a full-year autonomous Torch SURF run with Fortran references."""

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


SOIL_T_FIELDS = tuple(
    Field("soil_t", "SoilTemp", f"Soil temperature, layer {level + 1}", "K", level)
    for level in range(4)
)
SOIL_W_FIELDS = tuple(
    Field("soil_w", "SoilMoist", f"Soil water, layer {level + 1}", "kg m-2", level)
    for level in range(4)
)
SNOW_FIELDS = (
    Field("snow_w", "SWE", "Snow water equivalent", "kg m-2", snow_related=True),
    Field("snow_t", "SnowT", "Snow temperature", "K", snow_related=True),
    Field("snow_rho", "snowdens", "Snow density", "kg m-3", snow_related=True),
    Field("snow_alb", "SAlbedo", "Snow albedo", "1", snow_related=True),
)
FIELDS = SOIL_T_FIELDS + SOIL_W_FIELDS + SNOW_FIELDS


def paths_from_environment() -> tuple[Path, Path, Path, Path, Path]:
    try:
        torch_file = Path(os.environ["SURF_TORCH_ANNUAL"])
        fortran_10d = Path(os.environ["SURF_FORTRAN_10D"])
        fortran_final = Path(os.environ["SURF_FORTRAN_FINAL"])
        data_dir = Path(os.environ["SURF_DATA"])
        output_dir = Path(os.environ["SURF_COMPARE_OUTPUT"])
    except KeyError as exc:
        raise SystemExit(f"Missing required environment variable: {exc.args[0]}") from exc
    return torch_file, fortran_10d, fortran_final, data_dir, output_dir


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
        mask &= np.maximum(snow_fraction_candidate, snow_fraction_truth) > 1.0e-3
    elif domain != "all_columns":
        raise ValueError(f"unknown comparison domain: {domain}")
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
    swe_candidate = torch_at(torch, SNOW_FIELDS[0], torch_record)
    swe_truth = fortran_at(reference, SNOW_FIELDS[0], fortran_record)
    rho_candidate = torch_at(torch, SNOW_FIELDS[2], torch_record)
    rho_truth = fortran_at(reference, SNOW_FIELDS[2], fortran_record)
    snow_fraction_candidate = snow_fraction(swe_candidate, rho_candidate)
    snow_fraction_truth = snow_fraction(swe_truth, rho_truth)
    rows: list[dict[str, float | int | str]] = []
    for field in FIELDS:
        candidate = torch_at(torch, field, torch_record)
        truth = fortran_at(reference, field, fortran_record)
        domains = (
            ("all_columns", "snow_present", "snow_fraction_gt_1e-3")
            if field.snow_related else ("all_columns",)
        )
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
            maximum = int(row["max_index"])
            row.update({
                "torch_swe_at_max": float(swe_candidate[maximum]) if maximum >= 0 else float("nan"),
                "fortran_swe_at_max": float(swe_truth[maximum]) if maximum >= 0 else float("nan"),
                "torch_snow_fraction_at_max": float(snow_fraction_candidate[maximum]) if maximum >= 0 else float("nan"),
                "fortran_snow_fraction_at_max": float(snow_fraction_truth[maximum]) if maximum >= 0 else float("nan"),
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
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    groups = ((SOIL_T_FIELDS + SOIL_W_FIELDS, "Soil states"), (SNOW_FIELDS, "Snow states"))
    colors = plt.get_cmap("tab10").colors
    for row_index, (fields, title) in enumerate(groups):
        for column_index, (metric_name, ylabel) in enumerate((
            ("rmse", "RMSE (native unit)"),
            ("relative_rmse_percent", "Relative RMSE (%)"),
        )):
            axis = axes[row_index, column_index]
            for color, field in zip(colors, fields):
                domain = "snow_fraction_gt_1e-3" if field.label in {"snow_t", "snow_rho", "snow_alb"} else "all_columns"
                points = [point for point in rows if point["field"] == field.label and point["domain"] == domain]
                axis.semilogy(
                    [float(point["day"]) for point in points],
                    [float(point[metric_name]) for point in points],
                    color=color,
                    linewidth=1.1,
                    label=field.title,
                )
            axis.set(title=f"{title}: {ylabel}", ylabel=ylabel, xlim=(0.0, 365.0))
            axis.grid(alpha=0.25, which="both")
            axis.legend(fontsize=6.5, ncol=2, frameon=False)
    axes[1, 0].set_xlabel("Continuous integration day")
    axes[1, 1].set_xlabel("Continuous integration day")
    figure.suptitle("Autonomous offline SURF: Fortran-PyTorch error evolution", fontsize=13)
    figure.savefig(output_dir / "annual_autonomous_error_growth.png", dpi=260)
    plt.close(figure)


def plot_maps(
    torch: np.lib.npyio.NpzFile,
    reference: Dataset,
    torch_record: int,
    fortran_record: int,
    fields: tuple[Field, ...],
    lat: np.ndarray,
    lon: np.ndarray,
    output: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(len(fields), 3, figsize=(15, 2.8 * len(fields) + 1), constrained_layout=True)
    swe_candidate = torch_at(torch, SNOW_FIELDS[0], torch_record)
    swe_truth = fortran_at(reference, SNOW_FIELDS[0], fortran_record)
    rho_candidate = torch_at(torch, SNOW_FIELDS[2], torch_record)
    rho_truth = fortran_at(reference, SNOW_FIELDS[2], fortran_record)
    fraction_candidate = snow_fraction(swe_candidate, rho_candidate)
    fraction_truth = snow_fraction(swe_truth, rho_truth)
    for row_axes, field in zip(np.atleast_2d(axes), fields):
        candidate = torch_at(torch, field, torch_record)
        truth = fortran_at(reference, field, fortran_record)
        domain = "snow_fraction_gt_1e-3" if field.label in {"snow_t", "snow_rho", "snow_alb"} else "all_columns"
        mask = valid_mask(
            candidate, truth, swe_candidate, swe_truth, fraction_candidate, fraction_truth, domain,
        )
        candidate = np.where(mask, candidate, np.nan)
        truth = np.where(mask, truth, np.nan)
        difference = candidate - truth
        state_values = np.concatenate((candidate[np.isfinite(candidate)], truth[np.isfinite(truth)]))
        lo, hi = np.quantile(state_values, (0.01, 0.99))
        limit = max(float(np.nanquantile(np.abs(difference), 0.995)), np.finfo(np.float64).eps)
        panels = (
            (truth, "Fortran", "viridis", None, lo, hi),
            (candidate, "PyTorch", "viridis", None, lo, hi),
            (difference, "PyTorch - Fortran", "RdBu_r", TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), None, None),
        )
        for axis, (values, label, cmap, norm, vmin, vmax) in zip(row_axes, panels):
            x, y, image = grid(values, lat, lon)
            if norm is None:
                mesh = axis.pcolormesh(x, y, image, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                mesh = axis.pcolormesh(x, y, image, shading="nearest", cmap=cmap, norm=norm)
            axis.set(
                title=f"{field.title}: {label}",
                xlabel="Longitude (deg E)",
                ylabel="Latitude (deg N)",
                aspect="equal",
            )
            figure.colorbar(mesh, ax=axis, shrink=0.82, label=field.unit)
    figure.suptitle(title, fontsize=14)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    torch_path, fortran_10d_path, fortran_final_path, data_dir, output_dir = paths_from_environment()
    output_dir.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(np.load(data_dir / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(data_dir / "lon.npy", mmap_mode="r"), dtype=np.float64)
    with np.load(torch_path) as torch, Dataset(fortran_10d_path) as fortran_10d, Dataset(fortran_final_path) as fortran_final:
        torch_steps = np.asarray(torch["step"], dtype=int)
        if torch["soil_t"].shape[1] != lat.size or lon.size != lat.size:
            raise ValueError("latitude/longitude arrays do not match Torch column count")
        fortran_steps = np.asarray(fortran_10d.variables["timestp"][:], dtype=int)
        matches: list[tuple[int, int, int]] = []
        rows: list[dict[str, float | int | str]] = []
        for torch_record, step in enumerate(torch_steps):
            references = np.flatnonzero(fortran_steps == step)
            if references.size == 1:
                fortran_record = int(references[0])
                matches.append((torch_record, fortran_record, int(step)))
                day = float(fortran_10d.variables["time"][fortran_record]) / 86400.0
                rows.extend(field_rows(
                    torch, fortran_10d, torch_record, fortran_record, int(step), day, lat, lon,
                ))
        if not matches:
            raise ValueError("no Torch snapshots match the 10-day Fortran reference")
        final_steps = np.asarray(fortran_final.variables["timestp"][:], dtype=int)
        final_record = int(np.argmax(final_steps))
        final_step = int(final_steps[final_record])
        final_torch_records = np.flatnonzero(torch_steps == final_step)
        if final_torch_records.size != 1:
            raise ValueError(f"expected one Torch terminal snapshot at step {final_step}")
        terminal_torch_record = int(final_torch_records[0])
        terminal_day = float(fortran_final.variables["time"][final_record]) / 86400.0
        rows.extend(field_rows(
            torch, fortran_final, terminal_torch_record, final_record, final_step, terminal_day, lat, lon,
        ))

        with (output_dir / "annual_autonomous_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        terminal = [row for row in rows if int(row["step"]) == final_step]
        summary = {
            "torch_file": str(torch_path),
            "fortran_10d_file": str(fortran_10d_path),
            "fortran_final_file": str(fortran_final_path),
            "n_columns": int(lat.size),
            "time_series_steps": [int(step) for _, _, step in matches],
            "terminal_step": final_step,
            "terminal_day": terminal_day,
            "terminal_metrics": terminal,
        }
        (output_dir / "annual_autonomous_metrics.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True) + "\n"
        )
        plot_growth(rows, output_dir)
        plot_maps(
            torch, fortran_final, terminal_torch_record, final_record, SOIL_T_FIELDS,
            lat, lon, output_dir / "annual_autonomous_terminal_soil_temperature_maps.png",
            "Day 365 autonomous offline SURF: soil-temperature comparison",
        )
        plot_maps(
            torch, fortran_final, terminal_torch_record, final_record, SOIL_W_FIELDS,
            lat, lon, output_dir / "annual_autonomous_terminal_soil_water_maps.png",
            "Day 365 autonomous offline SURF: soil-water comparison",
        )
        plot_maps(
            torch, fortran_final, terminal_torch_record, final_record, SNOW_FIELDS,
            lat, lon, output_dir / "annual_autonomous_terminal_snow_maps.png",
            "Day 365 autonomous offline SURF: snow-state comparison",
        )
    print(json.dumps({
        "n_columns": int(lat.size),
        "time_series_records": len(matches),
        "terminal_step": final_step,
        "output": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
