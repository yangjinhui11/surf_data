#!/usr/bin/env python3
"""Gradient calibration of the day/night relaxation times of the closure.

modis_slab_asym.py sweeps tau_day x tau_night as a forward-only grid.  This
script makes the same two timescales trainable: the anomaly recurrence

    dT_{t+1} = beta_t dT_t + (1 - beta_t) (-lambda_s H_t),
    beta_t   = exp(-dt (w/tau_day + (1-w)/tau_night)),
    w        = SW / (SW + 50 W m-2)   (surface downward shortwave)

is kept in the autograd graph exactly as the one-parameter lambda_s
calibration of modis_structural_closure.py (slab mode): the build-up day
runs under no_grad, the forecast day under enable_grad, H_t enters
detached, and the loss is the cos-lat-weighted day+night MSE summed over
the ten calibration-window days on a strided column subset.  Adam acts on
(log tau_day, log tau_night); after every step both controls are clamped
to the sweep-stable range [tau_min, tau_max] h (night 1 h destabilizes the
held-out window, so the floor is the sweep's stable minimum).  After
training, prior (lambda_s = 0) and trained configurations are evaluated
full-domain on both windows with the sweep schema (per-day statistics,
aggregates, and local-solar-time composites).
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

import numpy as np
import torch
from netCDF4 import Dataset

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
SW_REF = 50.0          # W m-2, half-weight point of the day/night blend
N_LOCAL_BINS = 48      # 0.5-h bins of local solar time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--state-archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, default=0.05)
    parser.add_argument("--calibration-days", default="271,281,291,301,311,321,331,341,351,361")
    parser.add_argument("--validation-days", default="91,101,111,121,131,141,151,161,171,181")
    parser.add_argument("--column-stride", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.05,
                        help="Adam lr on log tau (both controls)")
    parser.add_argument("--init-tau-hours", type=float, default=None,
                        help="initial tau for both controls; default = the "
                             "tied closure value lambda_s*rho*cp*600 m")
    parser.add_argument("--init-tau-day-hours", type=float, default=None,
                        help="override initial tau_day (h)")
    parser.add_argument("--init-tau-night-hours", type=float, default=None,
                        help="override initial tau_night (h)")
    parser.add_argument("--tau-min-hours", type=float, default=2.0)
    parser.add_argument("--tau-min-day-hours", type=float, default=None,
                        help="override the day-side floor (h); default = "
                             "--tau-min-hours")
    parser.add_argument("--tau-min-night-hours", type=float, default=None,
                        help="override the night-side floor (h); default = "
                             "--tau-min-hours")
    parser.add_argument("--tau-max-hours", type=float, default=12.0)
    parser.add_argument("--lst-error-k", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260822)
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


def main() -> None:
    args = parse_args()
    calibration_days = [int(v) for v in args.calibration_days.split(",")]
    validation_days = [int(v) for v in args.validation_days.split(",")]
    if args.column_stride < 1 or args.iterations < 1:
        raise ValueError("--column-stride and --iterations must be positive")
    if args.tau_min_hours <= 0.0 or args.tau_max_hours <= args.tau_min_hours:
        raise ValueError("require 0 < tau_min < tau_max")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    lam_value = args.lambda_value
    tied_tau_s = lam_value * RHO_CP * 600.0
    init_tau_hours = args.init_tau_hours if args.init_tau_hours is not None \
        else tied_tau_s / 3600.0
    init_tau_day_hours = args.init_tau_day_hours if args.init_tau_day_hours is not None \
        else init_tau_hours
    init_tau_night_hours = args.init_tau_night_hours if args.init_tau_night_hours is not None \
        else init_tau_hours
    tau_min_day_s = (args.tau_min_day_hours if args.tau_min_day_hours is not None
                     else args.tau_min_hours) * 3600.0
    tau_min_night_s = (args.tau_min_night_hours if args.tau_min_night_hours is not None
                       else args.tau_min_hours) * 3600.0
    tau_max_s = args.tau_max_hours * 3600.0

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
    selected = slice(None, None, args.column_stride)
    column_count = len(range(0, nvalid, args.column_stride))

    full_lat = np.asarray(np.load(args.data / "lat.npy", mmap_mode="r"), dtype=np.float64)
    full_lon = np.asarray(np.load(args.data / "lon.npy", mmap_mode="r"), dtype=np.float64)
    lat, lon = full_lat[selected], full_lon[selected]
    forcing_subset = {name: np.load(args.data / f"{name}.npy", mmap_mode="r")[:, selected] for name in FORCING_NAMES}
    forcing_full = {name: np.load(args.data / f"{name}.npy", mmap_mode="r") for name in FORCING_NAMES}

    observations_calibration = load_observations(args.observations, calibration_days, selected)
    observations_validation = load_observations(args.observations, validation_days, selected)
    for observed in (observations_calibration, observations_validation):
        if not np.allclose(observed["lat"], lat, atol=1e-5) or not np.allclose(observed["lon"], lon, atol=1e-5):
            raise ValueError("MODIS and CMFD columns are not aligned")
    full_calibration_obs = load_observations(args.observations, calibration_days, slice(None))
    full_validation_obs = load_observations(args.observations, validation_days, slice(None))

    archive = np.load(args.state_archive, mmap_mode="r")
    archive_steps = np.asarray(archive["step"], dtype=np.int64)

    base_state_full, static_full = load_initial(
        args.case / "restartin.nc", nvalid, args.device, full_lat, full_lon,
    )
    base_state = subset_dataclass(base_state_full, selected, nvalid)
    static = subset_dataclass(static_full, selected, nvalid)

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

    # ------------------------------------------------------------------
    # Trainable control vector: log tau_day, log tau_night.
    # ------------------------------------------------------------------
    log_tau_day = torch.nn.Parameter(
        torch.tensor(math.log(init_tau_day_hours * 3600.0), dtype=torch.float64, device=args.device))
    log_tau_night = torch.nn.Parameter(
        torch.tensor(math.log(init_tau_night_hours * 3600.0), dtype=torch.float64, device=args.device))
    optimizer = torch.optim.Adam([log_tau_day, log_tau_night], lr=args.learning_rate)

    def clamp_parameters() -> None:
        with torch.no_grad():
            log_tau_day.clamp_(math.log(tau_min_day_s), math.log(tau_max_s))
            log_tau_night.clamp_(math.log(tau_min_night_s), math.log(tau_max_s))

    def control_values() -> list[float]:
        return [float(log_tau_day.exp().detach()) / 3600.0,
                float(log_tau_night.exp().detach()) / 3600.0]

    forcing_options = SimpleNamespace(
        shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
        shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
        stress_u=None, stress_v=None, vup=None,
    )
    sw_ref_t = torch.tensor(SW_REF, dtype=torch.float64, device=args.device)
    dt_const = torch.as_tensor(DT_SECONDS, dtype=torch.float64, device=args.device)

    def make_advance(step_static_base, column_lat, column_lon, forcing_arrays, use_closure: bool):
        """advance with the (differentiable) two-timescale recurrence.

        use_closure=False gives the prior (lambda_s = 0, no anomaly).  The
        taus are read from the live parameters so that training and the
        final full-domain evaluation share one code path; under an outer
        no_grad context the recurrence ops degrade to constants.
        """
        cache = {"heat": None, "dt_air": None}

        def advance(current_state, physical_step: int):
            fraction = model._tile_frac(step_static_base, config.n_tile, state=current_state)
            runtime = interpolated_runtime(
                forcing_arrays, physical_step, DT_SECONDS, column_lat, column_lon, args.device,
            )
            if use_closure:
                sw = runtime.swdown_W_m2[1].clamp(min=0.0)
                weight = sw / (sw + sw_ref_t)
                inv_tau = weight / log_tau_day.exp() + (1.0 - weight) / log_tau_night.exp()
                beta = torch.exp(-dt_const * inv_tau)
            emission = model._surface_emissivity(
                current_state, step_static_base, fraction, torch.float64, args.device,
            )
            step_static = replace(step_static_base, emis=emission)
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
            if use_closure:
                with torch.no_grad():
                    cache["heat"] = (fraction * model._last_pahfsti).sum(dim=1).detach().clone()
                # Keep the recurrence in the autograd graph: gradient reaches
                # the taus only through beta_t and the dT chain; H_t is
                # treated as exogenous within a step, as in the lambda_s
                # calibration.
                previous = cache["dt_air"] if cache["dt_air"] is not None \
                    else torch.zeros_like(cache["heat"])
                cache["dt_air"] = beta * previous + (1.0 - beta) * (-lam_value * cache["heat"])
            return new_state

        def reset_cache():
            cache["heat"] = None
            cache["dt_air"] = None

        return advance, reset_cache

    advance_subset, reset_subset = make_advance(static, lat, lon, forcing_subset, True)
    advance_full_prior, reset_full_prior = make_advance(static_full, full_lat, full_lon, forcing_full, False)
    advance_full_closed, reset_full_closed = make_advance(static_full, full_lat, full_lon, forcing_full, True)

    def day_predictions(day: int, with_gradient: bool, advance, reset, base, static_base,
                        width: int, observed_day: dict):
        archive_step = (day - 1) * STEPS_PER_DAY
        matching = np.flatnonzero(archive_steps == archive_step)
        if matching.size != 1:
            raise ValueError(f"state archive has no unique record for step {archive_step}")
        state = restore_state(copy.deepcopy(base), static_base, model, archive, int(matching[0]),
                              selected if width == column_count else slice(None),
                              args.device)
        reset()
        with torch.no_grad():
            for step in range(STEPS_PER_DAY):
                state = advance(state, archive_step + step)
        predictions = {period: torch.zeros(width, dtype=torch.float64, device=args.device) for period in PERIODS}
        period_steps = {period: target_steps(observed_day[period]["time"][0]) for period in PERIODS}
        context = torch.enable_grad() if with_gradient else torch.no_grad()
        with context:
            for step in range(STEPS_PER_DAY):
                state = advance(state, archive_step + STEPS_PER_DAY + step)
                completed = step + 1
                for period in PERIODS:
                    mask = torch.as_tensor(period_steps[period] == completed, dtype=torch.bool,
                                           device=args.device)
                    predictions[period] = torch.where(mask, state.radiative_skin_t, predictions[period])
        return predictions, period_steps

    area_weight = np.cos(np.deg2rad(lat))

    def day_loss_and_stats(predictions, observed_day, period_steps):
        stats = {}
        loss_terms = []
        for period in PERIODS:
            observation = observed_day[period]["lst"][0]
            valid = (
                observed_day[period]["strict"][0]
                & np.isfinite(observation)
                & (period_steps[period] >= 1)
            )
            count = int(valid.sum())
            if count == 0:
                stats[period] = {"n": 0, "bias_K": None, "rmse_K": None}
                continue
            valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=args.device)
            weights = area_weight[valid]
            weight_sum = float(weights.sum())
            residual = predictions[period][valid_tensor] - torch.as_tensor(
                observation[valid], dtype=torch.float64, device=args.device)
            term = (residual * residual * torch.as_tensor(
                weights / weight_sum, dtype=torch.float64, device=args.device)
            ).sum() / args.lst_error_k ** 2
            loss_terms.append(term)
            residual_np = residual.detach().cpu().numpy()
            stats[period] = {
                "n": count,
                "bias_K": float(residual_np.mean()),
                "rmse_K": float(np.sqrt((residual_np ** 2).mean())),
            }
        loss = torch.stack(loss_terms).sum() if loss_terms \
            else torch.zeros((), dtype=torch.float64, device=args.device)
        return loss, stats

    # ------------------------------------------------------------------
    # Adam optimization on the autumn-winter window (day+night loss).
    # ------------------------------------------------------------------
    history = []
    started = time.perf_counter()
    for iteration in range(args.iterations + 1):
        training = iteration > 0
        day_stats = {}
        total_loss = 0.0
        finite = True
        for group_day, day in enumerate(calibration_days):
            observed_day = single_day(observations_calibration, group_day)
            predictions, period_steps = day_predictions(
                day, training, advance_subset, reset_subset, base_state, static,
                column_count, observed_day,
            )
            loss_day, stats = day_loss_and_stats(predictions, observed_day, period_steps)
            if training:
                if torch.isfinite(loss_day):
                    loss_day.backward()
                else:
                    finite = False
            total_loss += float(loss_day.detach())
            day_stats[str(day)] = stats
        if training:
            grads_finite = all(
                p.grad is None or bool(torch.isfinite(p.grad).all())
                for p in (log_tau_day, log_tau_night)
            )
            if finite and grads_finite:
                optimizer.step()
            clamp_parameters()
            for parameter in (log_tau_day, log_tau_night):
                if parameter.grad is not None:
                    parameter.grad = None
        record = {
            "iteration": iteration,
            "loss": total_loss,
            "controls": control_values(),
            "days": day_stats,
            "stepped": bool(training and finite and grads_finite),
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "asym_train_history.json").write_text(json.dumps(history, indent=1) + "\n")
        aggregate = {}
        for period in PERIODS:
            entries = [d[period] for d in day_stats.values() if d[period]["rmse_K"] is not None]
            aggregate[period] = {
                "rmse_K": float(np.sqrt(np.mean([e["rmse_K"] ** 2 for e in entries]))) if entries else None,
                "bias_K": float(np.mean([e["bias_K"] for e in entries])) if entries else None,
            }
        print(json.dumps({"iteration": iteration, "loss": round(total_loss, 4),
                          "tau_day_h": round(record["controls"][0], 3),
                          "tau_night_h": round(record["controls"][1], 3),
                          "subset_rmse": {k: (round(v["rmse_K"], 3) if v["rmse_K"] is not None else None)
                                          for k, v in aggregate.items()},
                          "stepped": record["stepped"],
                          "elapsed_s": round(record["elapsed_s"], 1)}), flush=True)

    trained_tau = control_values()

    # ------------------------------------------------------------------
    # Post-optimization evaluation: prior vs trained, both windows, full
    # domain, sweep schema (per-day + aggregates + local-time composites).
    # ------------------------------------------------------------------
    def evaluate_config(advance, reset, label: str) -> dict:
        out = {}
        for window_name, days, observed in (
            ("calibration", calibration_days, full_calibration_obs),
            ("validation", validation_days, full_validation_obs),
        ):
            aggregate = {
                period: {"weighted_resid": 0.0, "sq": 0.0, "weight_sum": 0.0, "count": 0}
                for period in PERIODS
            }
            per_day = {}
            for group_day, day in enumerate(days):
                observed_day = single_day(observed, group_day)
                predictions, period_steps = day_predictions(
                    day, False, advance, reset, base_state_full, static_full,
                    nvalid, observed_day,
                )
                for period in PERIODS:
                    observation = observed_day[period]["lst"][0]
                    valid = observed_day[period]["strict"][0] & np.isfinite(observation) \
                        & (period_steps[period] >= 1)
                    if not valid.any():
                        per_day.setdefault(str(day), {})[period] = {
                            "n": 0, "bias_K": None, "rmse_K": None}
                        continue
                    valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=args.device)
                    weights = np.cos(np.deg2rad(full_lat))[valid]
                    residual = predictions[period][valid_tensor].cpu().numpy() - observation[valid]
                    aggregate[period]["weighted_resid"] += float((residual * weights).sum())
                    aggregate[period]["sq"] += float((residual ** 2 * weights).sum())
                    aggregate[period]["weight_sum"] += float(weights.sum())
                    aggregate[period]["count"] += int(residual.size)
                    per_day.setdefault(str(day), {})[period] = {
                        "n": int(valid.sum()),
                        "bias_K": float((residual * weights).sum() / weights.sum()),
                        "rmse_K": float(math.sqrt((residual ** 2 * weights).sum() / weights.sum())),
                    }
                print(json.dumps({"eval": label, "window": window_name, "day": day,
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
            summary = {
                period: {
                    "n": entry["count"],
                    "bias_K": entry["weighted_resid"] / entry["weight_sum"] if entry["count"] else None,
                    "rmse_K": math.sqrt(entry["sq"] / entry["weight_sum"]) if entry["count"] else None,
                }
                for period, entry in aggregate.items()
            }
            out[window_name] = {"summary": summary, "per_day": per_day}
            print(json.dumps({"eval": label, "window": window_name, "summary": summary}), flush=True)
        return out

    prior_eval = evaluate_config(advance_full_prior, reset_full_prior, "prior")
    trained_eval = evaluate_config(advance_full_closed, reset_full_closed, "trained")

    final = {
        "config": {
            "mode": "asym-tau-training",
            "control_names": ["tau_day_h", "tau_night_h"],
            "lambda_s": lam_value,
            "sw_ref_W_m2": SW_REF,
            "init_tau_hours": init_tau_hours,
            "tied_tau_hours": tied_tau_s / 3600.0,
            "tau_bounds_hours": [args.tau_min_hours, args.tau_max_hours],
            "tau_day_bounds_hours": [tau_min_day_s / 3600.0, args.tau_max_hours],
            "tau_night_bounds_hours": [tau_min_night_s / 3600.0, args.tau_max_hours],
            "calibration_days": calibration_days,
            "validation_days": validation_days,
            "column_stride": args.column_stride,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "lst_error_k": args.lst_error_k,
            "seed": args.seed,
            "case": str(args.case),
            "data": str(args.data),
            "observations": str(args.observations),
            "state_archive": str(args.state_archive),
        },
        "final_controls_hours": trained_tau,
        "prior": prior_eval,
        "trained": trained_eval,
    }
    args.output.write_text(json.dumps(final, indent=1) + "\n")
    print(json.dumps({"done": str(args.output),
                      "tau_day_h": trained_tau[0], "tau_night_h": trained_tau[1],
                      "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
