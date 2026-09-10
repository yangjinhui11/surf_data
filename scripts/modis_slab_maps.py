#!/usr/bin/env python3
"""Per-column and per-day evaluation of the slab closure vs the prior.

Runs the same spun-up, day-anchored protocol as modis_structural_closure.py
on the full domain (no optimization): for each lambda_s in --lambda-values
(default "0.0,0.05") it steps every calibration and validation day, samples
the grid-mean skin temperature at the MODIS view times, and records

  * per-column mean daytime/nighttime residual (for bias maps),
  * per-day area-weighted bias/RMSE (for per-day bar charts),
  * window-aggregate area-weighted bias/RMSE under two sampling rules:

      - protocol "30min": nearest 30-min step (the paper protocol, matches
        modis_structural_closure.py numbers),
      - protocol "3h": nearest 3-hourly record (matches the Fortran offline
        driver output cadence, for the cross-implementation appendix).

Outputs slab_maps.npz / slab_maps.json / slab_maps.png / slab_perday.png
next to --output.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

from modis_parameter_identifiability import (  # noqa: E402
    DT_SECONDS,
    FORCING_NAMES,
    PARAMETER_NAMES,
    PERIODS,
    STEPS_PER_DAY,
    restore_state,
    subset_dataclass,
    target_steps,
)

RHO_CP = 1.2 * 1005.7  # J m-3 K-1
THREE_H_STEPS = 6      # 30-min steps per 3-hourly Fortran record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--state-archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda-values", default="0.0,0.05")
    parser.add_argument("--calibration-days", default="271,281,291,301,311,321,331,341,351,361")
    parser.add_argument("--validation-days", default="91,101,111,121,131,141,151,161,171,181")
    parser.add_argument("--slab-depth", type=float, default=600.0)
    parser.add_argument("--tau-day-hours", type=float, default=None,
                        help="asymmetric closure: day relaxation time (h); requires "
                             "--tau-night-hours; default = tied tau = lambda_s*rho*cp*slab-depth")
    parser.add_argument("--tau-night-hours", type=float, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def load_observations(path: Path, days: list[int], column_slice: slice) -> dict:
    with Dataset(path) as observations:
        data = {}
        for period in PERIODS:
            data[period] = {
                "lst": np.asarray(observations[f"lst_{period}"][days, column_slice], dtype=np.float64),
                "time": np.asarray(observations[f"lst_{period}_view_time_utc"][days, column_slice], dtype=np.float64),
                "strict": np.asarray(observations[f"lst_{period}_strict"][days, column_slice]) == 1,
            }
        data["lat"] = np.asarray(observations["lat"][column_slice], dtype=np.float64)
        data["lon"] = np.asarray(observations["lon"][column_slice], dtype=np.float64)
    return data


def single_day(observed: dict, index: int) -> dict:
    return {
        period: {key: value[index: index + 1] for key, value in observed[period].items()}
        for period in PERIODS
    }


def nearest_3h_record(view_hours: np.ndarray) -> np.ndarray:
    """Index r in 0..7 of the record at 3(r+1) h nearest each view time."""
    return np.clip(np.rint(view_hours / 3.0).astype(int) - 1, 0, STEPS_PER_DAY // THREE_H_STEPS - 1)


def main() -> None:
    args = parse_args()
    calibration_days = [int(v) for v in args.calibration_days.split(",")]
    validation_days = [int(v) for v in args.validation_days.split(",")]
    lambda_values = [float(v) for v in args.lambda_values.split(",")]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    os.environ["SURF_CASE"] = str(args.case)
    os.environ["SURF_CODE"] = str(args.source)
    os.environ["SURF_DATA"] = str(args.data)
    sys.path[:0] = [str(args.source), str(args.runner)]
    from diagnostic_scripts_compare_realcol import make_forcing  # noqa: E402
    from replay_full_domain_controlled import load_initial  # noqa: E402
    from run_full_domain_offline import interpolated_runtime  # noqa: E402
    from surf_pytorch import LandSurface, SurfConfig  # noqa: E402

    metadata = json.loads((args.case / "metadata.json").read_text())
    nvalid = int(metadata["nvalid"])

    full_lat = np.asarray(np.load(args.data / "lat.npy", mmap_mode="r"), dtype=np.float64)
    full_lon = np.asarray(np.load(args.data / "lon.npy", mmap_mode="r"), dtype=np.float64)
    forcing_full = {name: np.load(args.data / f"{name}.npy", mmap_mode="r") for name in FORCING_NAMES}

    full_calibration_obs = load_observations(args.observations, calibration_days, slice(None))
    full_validation_obs = load_observations(args.observations, validation_days, slice(None))

    archive = np.load(args.state_archive, mmap_mode="r")
    archive_steps = np.asarray(archive["step"], dtype=np.int64)

    base_state_full, static_full = load_initial(
        args.case / "restartin.nc", nvalid, args.device, full_lat, full_lon,
    )

    config = SurfConfig(
        dtype=torch.float64,
        device=args.device,
        differentiable=True,
        validate=False,
        use_farquhar=True,
        use_ags=False,
        levgen=True,
        lelwtl=True,
        lelaiv=False,
        vup_boundary_mode="offline",
        learnable_veg=PARAMETER_NAMES,
        learnable_soil=("lambdadry", "lamsat1", "rcsoil"),
    )
    model = LandSurface(config).to(args.device)

    forcing_options = SimpleNamespace(
        shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
        shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
        stress_u=None, stress_v=None, vup=None,
    )

    asym = (args.tau_day_hours is not None and args.tau_night_hours is not None)
    if (args.tau_day_hours is None) != (args.tau_night_hours is None):
        raise ValueError("--tau-day-hours and --tau-night-hours must be given together")
    sw_ref_t = torch.tensor(50.0, dtype=torch.float64, device=args.device)
    dt_const = torch.as_tensor(DT_SECONDS, dtype=torch.float64, device=args.device)
    tau_day_t = tau_night_t = None
    if asym:
        tau_day_t = torch.tensor(args.tau_day_hours * 3600.0, dtype=torch.float64, device=args.device)
        tau_night_t = torch.tensor(args.tau_night_hours * 3600.0, dtype=torch.float64, device=args.device)

    def make_advance(lam_value: float):
        scalar_beta = None
        if lam_value > 1e-9 and not asym:
            scalar_beta = math.exp(-DT_SECONDS / (lam_value * RHO_CP * args.slab_depth))
        cache = {"heat": None, "dt_air": None}

        def advance(current_state, physical_step: int):
            fraction = model._tile_frac(static_full, config.n_tile, state=current_state)
            runtime = interpolated_runtime(
                forcing_full, physical_step, DT_SECONDS, full_lat, full_lon, args.device,
            )
            emission = model._surface_emissivity(
                current_state, static_full, fraction, torch.float64, args.device,
            )
            step_static = replace(static_full, emis=emission)
            forcing = make_forcing(runtime, 0, current_state, step_static, forcing_options, tile_frac=fraction)
            precipitation = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
            forcing = replace(
                forcing,
                rain=precipitation * (1.0 - runtime.precip_snow_phase),
                snow=precipitation * runtime.precip_snow_phase,
            )
            if cache["dt_air"] is not None:
                forcing = replace(forcing, t=forcing.t + cache["dt_air"])
            new_state = model(current_state, forcing, step_static)[0]
            with torch.no_grad():
                cache["heat"] = (fraction * model._last_pahfsti).sum(dim=1).detach().clone()
                if lam_value > 1e-9:
                    if scalar_beta is not None:
                        beta = scalar_beta
                    else:
                        sw = runtime.swdown_W_m2[1].clamp(min=0.0)
                        weight = sw / (sw + sw_ref_t)
                        beta = torch.exp(-dt_const * (weight / tau_day_t
                                                      + (1.0 - weight) / tau_night_t))
                    previous = cache["dt_air"] if cache["dt_air"] is not None \
                        else torch.zeros_like(cache["heat"])
                    cache["dt_air"] = beta * previous + (1.0 - beta) * (-lam_value * cache["heat"])
            return new_state

        def reset():
            cache["heat"] = None
            cache["dt_air"] = None

        return advance, reset

    def day_predictions(day: int, advance, reset, observed_day: dict):
        """Predictions at both sampling rules for obs day D (full domain)."""
        archive_step = (day - 1) * STEPS_PER_DAY
        matching = np.flatnonzero(archive_steps == archive_step)
        if matching.size != 1:
            raise ValueError(f"state archive has no unique record for step {archive_step}")
        state = restore_state(copy.deepcopy(base_state_full), static_full, model, archive, int(matching[0]),
                              slice(None), args.device)
        reset()
        with torch.no_grad():
            for step in range(STEPS_PER_DAY):
                state = advance(state, archive_step + step)
            predictions = {
                period: {
                    "30min": torch.zeros(nvalid, dtype=torch.float64, device=args.device),
                    "3h": torch.zeros(nvalid, dtype=torch.float64, device=args.device),
                }
                for period in PERIODS
            }
            period_steps = {period: target_steps(observed_day[period]["time"][0]) for period in PERIODS}
            period_records = {period: nearest_3h_record(observed_day[period]["time"][0]) for period in PERIODS}
            for step in range(STEPS_PER_DAY):
                state = advance(state, archive_step + STEPS_PER_DAY + step)
                completed = step + 1
                for period in PERIODS:
                    mask = torch.as_tensor(period_steps[period] == completed, dtype=torch.bool, device=args.device)
                    predictions[period]["30min"] = torch.where(mask, state.radiative_skin_t, predictions[period]["30min"])
                if completed % THREE_H_STEPS == 0:
                    record = completed // THREE_H_STEPS - 1  # 0..7 at 3(r+1) h
                    for period in PERIODS:
                        mask = torch.as_tensor(period_records[period] == record, dtype=torch.bool, device=args.device)
                        predictions[period]["3h"] = torch.where(mask, state.radiative_skin_t, predictions[period]["3h"])
        return predictions, period_steps, period_records

    area_weight = np.cos(np.deg2rad(full_lat))
    windows = {
        "calibration": (calibration_days, full_calibration_obs),
        "validation": (validation_days, full_validation_obs),
    }

    result = {"lambda_values": lambda_values, "slab_depth_m": args.slab_depth,
              "tau_day_h": args.tau_day_hours, "tau_night_h": args.tau_night_hours,
              "windows": {name: days for name, (days, _) in windows.items()}}
    per_column = {}
    per_day = {}
    started = time.perf_counter()

    for lam_value in lambda_values:
        advance, reset = make_advance(lam_value)
        lam_key = f"lambda_{lam_value:.4f}"
        per_column[lam_key] = {}
        per_day[lam_key] = {}
        for window_name, (days, observed) in windows.items():
            per_column[lam_key][window_name] = {
                period: {"resid_sum": np.zeros(nvalid), "count": np.zeros(nvalid, dtype=int)}
                for period in PERIODS
            }
            per_day[lam_key][window_name] = {}
            aggregate = {
                (period, protocol): {"weighted_resid": 0.0, "sq": 0.0, "weight_sum": 0.0, "count": 0}
                for period in PERIODS for protocol in ("30min", "3h")
            }
            for group_day, day in enumerate(days):
                observed_day = single_day(observed, group_day)
                predictions, period_steps, period_records = day_predictions(day, advance, reset, observed_day)
                day_entry = {}
                for period in PERIODS:
                    observation = observed_day[period]["lst"][0]
                    valid = observed_day[period]["strict"][0] & np.isfinite(observation) & (period_steps[period] >= 1)
                    if not valid.any():
                        day_entry[period] = {"n": 0, "bias_K": None, "rmse_K": None}
                        continue
                    valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=args.device)
                    weights = area_weight[valid]
                    per_column[lam_key][window_name][period]["resid_sum"][valid] += \
                        (predictions[period]["30min"][valid_tensor].cpu().numpy() - observation[valid])
                    per_column[lam_key][window_name][period]["count"][valid] += 1
                    day_stats = {}
                    for protocol in ("30min", "3h"):
                        residual = predictions[period][protocol][valid_tensor].cpu().numpy() - observation[valid]
                        aggregate[(period, protocol)]["weighted_resid"] += float((residual * weights).sum())
                        aggregate[(period, protocol)]["sq"] += float((residual ** 2 * weights).sum())
                        aggregate[(period, protocol)]["weight_sum"] += float(weights.sum())
                        aggregate[(period, protocol)]["count"] += int(residual.size)
                        day_stats[protocol] = {
                            "bias_K": float((residual * weights).sum() / weights.sum()),
                            "rmse_K": float(math.sqrt((residual ** 2 * weights).sum() / weights.sum())),
                        }
                    day_entry[period] = {"n": int(valid.sum()), **{f"bias_K_{p}": day_stats[p]["bias_K"] for p in day_stats},
                                         **{f"rmse_K_{p}": day_stats[p]["rmse_K"] for p in day_stats}}
                per_day[lam_key][window_name][str(day)] = day_entry
                print(json.dumps({"lambda": lam_value, "window": window_name, "day": day,
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
            result.setdefault(lam_key, {})[window_name] = {
                f"{period}_{protocol}": {
                    "n": entry["count"],
                    "bias_K": entry["weighted_resid"] / entry["weight_sum"] if entry["count"] else None,
                    "rmse_K": math.sqrt(entry["sq"] / entry["weight_sum"]) if entry["count"] else None,
                }
                for (period, protocol), entry in aggregate.items()
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result["per_day"] = per_day
    (args.output.parent / "slab_maps.json").write_text(json.dumps(result, indent=1) + "\n")

    npz_payload = {"lat": full_lat, "lon": full_lon}
    for lam_key in per_column:
        for window_name in per_column[lam_key]:
            for period in PERIODS:
                entry = per_column[lam_key][window_name][period]
                mean = np.full(nvalid, np.nan)
                nonzero = entry["count"] > 0
                mean[nonzero] = entry["resid_sum"][nonzero] / entry["count"][nonzero]
                npz_payload[f"{lam_key}__{window_name}__{period}"] = mean
                npz_payload[f"{lam_key}__{window_name}__{period}__count"] = entry["count"]
    np.savez_compressed(args.output.parent / "slab_maps.npz", **npz_payload)

    # ------------------------------------------------------------------
    # Figure 1: daytime bias maps, prior vs slab, both windows.
    # ------------------------------------------------------------------
    lam_prior, lam_slab = (f"lambda_{v:.4f}" for v in lambda_values)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for row, window_name in enumerate(("calibration", "validation")):
        for col, (label, lam_key) in enumerate((("SURF", lam_prior), ("SURF-Opt", lam_slab))):
            axis = axes[row, col]
            field = npz_payload[f"{lam_key}__{window_name}__day"]
            scatter = axis.scatter(full_lon, full_lat, c=field, cmap="RdBu_r", vmin=-12, vmax=12,
                                   s=0.25, linewidths=0)
            finite = np.isfinite(field)
            weights = area_weight[finite]
            weighted_mean = float((field[finite] * weights).sum() / weights.sum())
            axis.set_title(f"{window_name} window -- {label}: column mean day bias "
                           f"(mean {weighted_mean:+.2f} K)", fontsize=10)
            axis.set_xlabel("lon", fontsize=8)
            axis.set_ylabel("lat", fontsize=8)
            axis.tick_params(labelsize=7)
    figure.colorbar(scatter, ax=axes, label="daytime bias, model - MODIS (K)", shrink=0.9)
    figure.suptitle("Daytime LST bias vs MODIS: SURF vs SURF-Opt "
                    f"(one-parameter surface-layer closure, lambda_s={lambda_values[-1]} K/(W m$^{{-2}}$))", fontsize=11)
    figure.savefig(args.output.parent / "slab_maps.png", dpi=260)
    plt.close(figure)

    # ------------------------------------------------------------------
    # Figure 2: per-day area-weighted day bias and RMSE, prior vs slab.
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for axis, metric in zip(axes, ("bias", "rmse")):
        tick_position, tick_label, position = [], [], 0.0
        for window_name in ("calibration", "validation"):
            days = windows[window_name][0]
            group_width = 0.9
            for day_index, day in enumerate(days):
                center = position + day_index
                for offset, lam_key, color in ((-0.21, lam_prior, "0.6"), (0.21, lam_slab, "tab:blue")):
                    entry = per_day[lam_key][window_name][str(day)]["day"]
                    value = entry.get(f"{metric}_K_30min")
                    if value is not None:
                        axis.bar(center + offset, value, width=0.38, color=color,
                                 label="SURF" if (day_index == 0 and offset < 0 and window_name == "calibration")
                                 else ("SURF-Opt" if (day_index == 0 and offset > 0 and window_name == "calibration") else None))
                tick_position.append(center)
                tick_label.append(str(day))
            position = len(tick_position) + 1.0
        if metric == "bias":
            axis.axhline(0.0, color="k", linewidth=0.8)
        axis.set_xticks(tick_position)
        axis.set_xticklabels(tick_label, rotation=45, fontsize=6)
        axis.set_xlabel("day of year 2018 (left: calibration, right: validation)", fontsize=9)
        axis.set_ylabel(f"daytime {metric} (K)", fontsize=9)
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=8)
    figure.suptitle("Daytime bias and RMSE vs MODIS per evaluation day: SURF vs SURF-Opt", fontsize=11)
    figure.savefig(args.output.parent / "slab_perday.png", dpi=260)
    plt.close(figure)
    print(json.dumps({"done": str(args.output.parent / "slab_maps.json"),
                      "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
