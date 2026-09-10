#!/usr/bin/env python3
"""Compare SURF observation-operator output with MODIS Terra CMG products."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date


def read_variable(dataset: Dataset, name: str) -> np.ndarray:
    values = dataset[name][:]
    fill = np.nan if np.issubdtype(values.dtype, np.floating) else 255
    return np.asarray(np.ma.filled(values, fill))


def continuous_metrics(
    model: np.ndarray, observation: np.ndarray, mask: np.ndarray, weights: np.ndarray,
) -> dict[str, float | int]:
    valid = mask & np.isfinite(model) & np.isfinite(observation)
    count = int(valid.sum())
    if count == 0:
        return {
            "count": 0, "model_mean": float("nan"), "observation_mean": float("nan"),
            "bias": float("nan"), "mae": float("nan"), "rmse": float("nan"),
            "correlation": float("nan"),
        }
    candidate, truth = model[valid].astype(np.float64), observation[valid].astype(np.float64)
    sample_weights = np.broadcast_to(weights, model.shape)[valid].astype(np.float64)
    sample_weights /= sample_weights.sum()
    difference = candidate - truth
    candidate_anomaly = candidate - np.sum(sample_weights * candidate)
    truth_anomaly = truth - np.sum(sample_weights * truth)
    covariance = np.sum(sample_weights * candidate_anomaly * truth_anomaly)
    variance_product = np.sum(sample_weights * candidate_anomaly**2) * np.sum(sample_weights * truth_anomaly**2)
    correlation = float(covariance / np.sqrt(variance_product)) if variance_product > 0 else float("nan")
    return {
        "count": count,
        "model_mean": float(np.sum(sample_weights * candidate)),
        "observation_mean": float(np.sum(sample_weights * truth)),
        "bias": float(np.sum(sample_weights * difference)),
        "mae": float(np.sum(sample_weights * np.abs(difference))),
        "rmse": float(np.sqrt(np.sum(sample_weights * difference**2))),
        "correlation": correlation,
    }


def snow_detection_metrics(
    model: np.ndarray, observation: np.ndarray, mask: np.ndarray, weights: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    valid = mask & np.isfinite(model) & np.isfinite(observation)
    predicted = model[valid] >= threshold
    observed = observation[valid] >= threshold
    sample_weights = np.broadcast_to(weights, model.shape)[valid].astype(np.float64)
    hits = int(np.sum(predicted & observed))
    misses = int(np.sum(~predicted & observed))
    false_alarms = int(np.sum(predicted & ~observed))
    weighted_hits = float(np.sum(sample_weights[predicted & observed]))
    weighted_misses = float(np.sum(sample_weights[~predicted & observed]))
    weighted_false_alarms = float(np.sum(sample_weights[predicted & ~observed]))

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "count": int(valid.sum()),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "area_weighted_probability_of_detection": ratio(weighted_hits, weighted_hits + weighted_misses),
        "area_weighted_false_alarm_ratio": ratio(weighted_false_alarms, weighted_hits + weighted_false_alarms),
        "area_weighted_critical_success_index": ratio(weighted_hits, weighted_hits + weighted_misses + weighted_false_alarms),
    }


def daily_stat(
    model: np.ndarray, observation: np.ndarray, mask: np.ndarray, weights: np.ndarray,
    statistic: str,
) -> np.ndarray:
    result = np.full(model.shape[0], np.nan, dtype=np.float64)
    for day in range(model.shape[0]):
        metrics = continuous_metrics(model[day], observation[day], mask[day], weights)
        result[day] = metrics[statistic]
    return result


def column_bias(model: np.ndarray, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    difference = np.where(mask & np.isfinite(model) & np.isfinite(observation), model - observation, np.nan)
    count = np.sum(np.isfinite(difference), axis=0)
    total = np.nansum(difference, axis=0)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def surface_air_contrast(
    tair: np.ndarray, view_time_utc: np.ndarray, model: np.ndarray,
    observation: np.ndarray, mask: np.ndarray, weights: np.ndarray,
) -> dict[str, float | int]:
    """Area-weighted LST-minus-air-temperature contrast at satellite times."""
    forcing_interval = 10_800.0
    nrecord, ncolumn = tair.shape
    if model.shape[1] != ncolumn:
        raise ValueError("CMFD air-temperature columns do not match the validation domain")
    count = 0
    weight_sum = 0.0
    model_sum = 0.0
    observation_sum = 0.0
    columns = np.arange(ncolumn)
    for day in range(model.shape[0]):
        valid = mask[day] & np.isfinite(view_time_utc[day]) & np.isfinite(model[day]) & np.isfinite(observation[day])
        if not np.any(valid):
            continue
        seconds = day * 86_400.0 + view_time_utc[day, valid].astype(np.float64) * 3600.0
        record = (seconds // forcing_interval).astype(np.int64) % nrecord
        next_record = (record + 1) % nrecord
        alpha = (seconds - np.floor(seconds / forcing_interval) * forcing_interval) / forcing_interval
        selected_columns = columns[valid]
        air = (
            np.asarray(tair[record, selected_columns], dtype=np.float32).astype(np.float64)
            + alpha * (
                np.asarray(tair[next_record, selected_columns], dtype=np.float32).astype(np.float64)
                - np.asarray(tair[record, selected_columns], dtype=np.float32).astype(np.float64)
            )
        )
        sample_weights = weights[valid]
        weight_sum += float(sample_weights.sum())
        model_sum += float(np.sum(sample_weights * (model[day, valid] - air)))
        observation_sum += float(np.sum(sample_weights * (observation[day, valid] - air)))
        count += int(valid.sum())
    return {
        "count": count,
        "model_skin_minus_air_K": model_sum / weight_sum,
        "observed_lst_minus_air_K": observation_sum / weight_sum,
    }


def running_mean(values: np.ndarray, width: int = 7) -> np.ndarray:
    finite = np.isfinite(values)
    numerator = np.convolve(np.where(finite, values, 0.0), np.ones(width), mode="same")
    denominator = np.convolve(finite.astype(float), np.ones(width), mode="same")
    return np.divide(numerator, denominator, out=np.full(values.shape, np.nan), where=denominator > 0)


def plot_timeseries(
    output: Path, dates: np.ndarray, fields: dict[str, np.ndarray],
    monthly_rows: list[dict[str, float | int]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8), constrained_layout=True)
    months = np.arange(1, len(monthly_rows) + 1)
    axes[0, 0].plot(months, [row["lst_day_bias_K"] for row in monthly_rows], "o-", color="#b33b2e", linewidth=1.1, label="day")
    axes[0, 0].plot(months, [row["lst_night_bias_K"] for row in monthly_rows], "o-", color="#315b8a", linewidth=1.1, label="night")
    axes[0, 0].axhline(0, color="0.25", linewidth=0.6)
    axes[0, 0].set_ylabel("LST bias (K)")
    axes[0, 0].legend(frameon=False, ncol=2)

    axes[0, 1].plot(months, [row["lst_day_rmse_K"] for row in monthly_rows], "o-", color="#b33b2e", linewidth=1.1, label="day")
    axes[0, 1].plot(months, [row["lst_night_rmse_K"] for row in monthly_rows], "o-", color="#315b8a", linewidth=1.1, label="night")
    axes[0, 1].set_ylabel("LST RMSE (K)")

    axes[1, 0].plot(dates, running_mean(fields["snow_obs_area"]), color="#315b8a", linewidth=1.0, label="MODIS")
    axes[1, 0].plot(dates, running_mean(fields["snow_model_area"]), color="#b33b2e", linewidth=1.0, label="SURF")
    axes[1, 0].set_ylabel("Snow-covered fraction")
    axes[1, 0].legend(frameon=False, ncol=2)

    axes[1, 1].plot(months, [row["snow_rmse"] for row in monthly_rows], "o-", color="#4b7f52", linewidth=1.1, label="RMSE")
    axes[1, 1].set_ylabel("Snow-fraction RMSE")
    score_axis = axes[1, 1].twinx()
    score_axis.plot(months, [row["snow_critical_success_index"] for row in monthly_rows], "s--", color="#6b4c8a", linewidth=1.0, label="CSI")
    score_axis.set_ylabel("Critical success index")
    score_axis.set_ylim(0, 1)
    for axis in axes[0, :]:
        axis.set_xticks(months)
        axis.set_xlabel("Month of 2018")
    axes[1, 1].set_xticks(months)
    axes[1, 1].set_xlabel("Month of 2018")
    for axis in axes.flat:
        axis.grid(color="0.88", linewidth=0.5)
    axes[1, 0].set_xlabel("2018")
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_maps(output: Path, lon: np.ndarray, lat: np.ndarray, maps: list[tuple[str, np.ndarray, str, tuple[float, float]]]) -> None:
    fig, axes = plt.subplots(1, len(maps), figsize=(12.2, 3.8), constrained_layout=True)
    for axis, (title, values, cmap, limits) in zip(axes, maps):
        image = axis.scatter(lon, lat, c=values, s=1.4, marker="s", linewidths=0, cmap=cmap, vmin=limits[0], vmax=limits[1], rasterized=True)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
        axis.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
        fig.colorbar(image, ax=axis, orientation="horizontal", pad=0.12, shrink=0.86)
    fig.savefig(output, dpi=260)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forcing-data", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with Dataset(args.observations) as obs, Dataset(args.model) as model_ds:
        lat, lon = read_variable(obs, "lat"), read_variable(obs, "lon")
        model_lat, model_lon = read_variable(model_ds, "lat"), read_variable(model_ds, "lon")
        if not np.allclose(lat, model_lat, atol=1.0e-5) or not np.allclose(lon, model_lon, atol=1.0e-5):
            raise ValueError("model and observation columns are not aligned")
        time_var = obs["time"]
        dates = np.array([
            np.datetime64(f"{value.year:04d}-{value.month:02d}-{value.day:02d}")
            for value in num2date(time_var[:], time_var.units, time_var.calendar)
        ])
        observed = {name: read_variable(obs, name) for name in (
            "lst_day", "lst_night", "lst_day_view_time_utc", "lst_night_view_time_utc",
            "lst_day_good", "lst_night_good",
            "lst_day_strict", "lst_night_strict", "snow_fraction", "snow_good",
        )}
        modeled = {name: read_variable(model_ds, name) for name in ("lst_day", "lst_night", "snow_fraction")}

    if modeled["lst_day"].shape != observed["lst_day"].shape:
        raise ValueError("model and observation time/column dimensions differ")
    masks = {
        "lst_day_strict": observed["lst_day_strict"] == 1,
        "lst_night_strict": observed["lst_night_strict"] == 1,
        "lst_day_good": observed["lst_day_good"] == 1,
        "lst_night_good": observed["lst_night_good"] == 1,
        "snow_good": observed["snow_good"] == 1,
    }
    area_weights = np.cos(np.deg2rad(lat)).astype(np.float64)

    def period_metrics(selection: np.ndarray) -> dict[str, dict[str, float | int]]:
        subset_model = {name: values[selection] for name, values in modeled.items()}
        subset_obs = {name: values[selection] for name, values in observed.items()}
        subset_mask = {name: values[selection] for name, values in masks.items()}
        return {
            "lst_day_strict": continuous_metrics(subset_model["lst_day"], subset_obs["lst_day"], subset_mask["lst_day_strict"], area_weights),
            "lst_night_strict": continuous_metrics(subset_model["lst_night"], subset_obs["lst_night"], subset_mask["lst_night_strict"], area_weights),
            "lst_day_good": continuous_metrics(subset_model["lst_day"], subset_obs["lst_day"], subset_mask["lst_day_good"], area_weights),
            "lst_night_good": continuous_metrics(subset_model["lst_night"], subset_obs["lst_night"], subset_mask["lst_night_good"], area_weights),
            "snow_fraction": continuous_metrics(subset_model["snow_fraction"], subset_obs["snow_fraction"], subset_mask["snow_good"], area_weights),
            "snow_detection_0.5": snow_detection_metrics(subset_model["snow_fraction"], subset_obs["snow_fraction"], subset_mask["snow_good"], area_weights),
        }

    day_index = np.arange(dates.size)
    month_index = np.array([int(str(value)[5:7]) for value in dates])
    periods = {
        "full_year": np.ones(dates.size, dtype=bool),
        "day_31_to_365": day_index >= 30,
        "july_to_december": month_index >= 7,
    }
    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "observations": str(args.observations),
        "model": str(args.model),
        "spatial_weighting": "cos(latitude)",
        "periods": {name: period_metrics(selection) for name, selection in periods.items()},
    }
    if args.forcing_data is not None:
        tair = np.load(args.forcing_data / "tair_K.npy", mmap_mode="r")
        metrics["surface_air_contrast"] = {
            f"{period}_{quality}": surface_air_contrast(
                tair,
                observed[f"lst_{period}_view_time_utc"],
                modeled[f"lst_{period}"],
                observed[f"lst_{period}"],
                masks[f"lst_{period}_{quality}"],
                area_weights,
            )
            for quality in ("strict", "good")
            for period in ("day", "night")
        }

    day_fields = {
        "lst_day_bias": daily_stat(modeled["lst_day"], observed["lst_day"], masks["lst_day_strict"], area_weights, "bias"),
        "lst_night_bias": daily_stat(modeled["lst_night"], observed["lst_night"], masks["lst_night_strict"], area_weights, "bias"),
        "lst_day_rmse": daily_stat(modeled["lst_day"], observed["lst_day"], masks["lst_day_strict"], area_weights, "rmse"),
        "lst_night_rmse": daily_stat(modeled["lst_night"], observed["lst_night"], masks["lst_night_strict"], area_weights, "rmse"),
        "snow_rmse": daily_stat(modeled["snow_fraction"], observed["snow_fraction"], masks["snow_good"], area_weights, "rmse"),
    }
    valid_snow = masks["snow_good"] & np.isfinite(observed["snow_fraction"])
    area_weights_2d = np.broadcast_to(area_weights, valid_snow.shape)
    day_fields["snow_obs_area"] = np.divide(
        np.nansum(np.where(valid_snow, observed["snow_fraction"] * area_weights_2d, np.nan), axis=1),
        np.sum(np.where(valid_snow, area_weights_2d, 0.0), axis=1),
        out=np.full(valid_snow.shape[0], np.nan), where=np.sum(valid_snow, axis=1) > 0,
    )
    valid_model_snow = valid_snow & np.isfinite(modeled["snow_fraction"])
    day_fields["snow_model_area"] = np.divide(
        np.nansum(np.where(valid_model_snow, modeled["snow_fraction"] * area_weights_2d, np.nan), axis=1),
        np.sum(np.where(valid_model_snow, area_weights_2d, 0.0), axis=1),
        out=np.full(valid_snow.shape[0], np.nan), where=np.sum(valid_model_snow, axis=1) > 0,
    )

    with (args.output_dir / "modis_validation_daily.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        names = list(day_fields)
        writer.writerow(["date", *names])
        for index, day in enumerate(dates):
            writer.writerow([str(day), *(day_fields[name][index] for name in names)])
    monthly_rows = []
    for month in range(1, 13):
        selection = month_index == month
        if not np.any(selection):
            continue
        month_metrics = period_metrics(selection)
        monthly_rows.append({
            "month": month,
            "lst_day_count": month_metrics["lst_day_strict"]["count"],
            "lst_day_bias_K": month_metrics["lst_day_strict"]["bias"],
            "lst_day_rmse_K": month_metrics["lst_day_strict"]["rmse"],
            "lst_day_correlation": month_metrics["lst_day_strict"]["correlation"],
            "lst_night_count": month_metrics["lst_night_strict"]["count"],
            "lst_night_bias_K": month_metrics["lst_night_strict"]["bias"],
            "lst_night_rmse_K": month_metrics["lst_night_strict"]["rmse"],
            "lst_night_correlation": month_metrics["lst_night_strict"]["correlation"],
            "snow_count": month_metrics["snow_fraction"]["count"],
            "snow_bias": month_metrics["snow_fraction"]["bias"],
            "snow_rmse": month_metrics["snow_fraction"]["rmse"],
            "snow_correlation": month_metrics["snow_fraction"]["correlation"],
            "snow_probability_of_detection": month_metrics["snow_detection_0.5"]["area_weighted_probability_of_detection"],
            "snow_false_alarm_ratio": month_metrics["snow_detection_0.5"]["area_weighted_false_alarm_ratio"],
            "snow_critical_success_index": month_metrics["snow_detection_0.5"]["area_weighted_critical_success_index"],
        })
    with (args.output_dir / "modis_validation_monthly.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(monthly_rows[0]))
        writer.writeheader()
        writer.writerows(monthly_rows)
    (args.output_dir / "modis_validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_timeseries(args.output_dir / "modis_validation_timeseries.png", dates, day_fields, monthly_rows)

    map_values = [
        ("Daytime LST bias (K)", column_bias(modeled["lst_day"], observed["lst_day"], masks["lst_day_strict"]), "RdBu_r", (-8, 8)),
        ("Nighttime LST bias (K)", column_bias(modeled["lst_night"], observed["lst_night"], masks["lst_night_strict"]), "RdBu_r", (-8, 8)),
        ("Snow-fraction bias", column_bias(modeled["snow_fraction"], observed["snow_fraction"], masks["snow_good"]), "BrBG", (-0.5, 0.5)),
    ]
    plot_maps(args.output_dir / "modis_validation_maps.png", lon, lat, map_values)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
