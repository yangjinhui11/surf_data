#!/usr/bin/env python3
"""Structural closure experiments on the MODIS daytime cold bias.

Two AD-driven experiments that go *beyond multiplicative parameter scales*
(modis_calibration_heldout.py absorbs only ~0.5 K of the ~5.5 K daytime bias,
so the ceiling is structural, not parametric):

``attribution``
    Pseudo-forcing control vector: eight additive T_air offsets, one per
    local 3-hour bin, plus multiplicative wind and shortwave scales.
    Optimizing these on the same spun-up calibration window answers
    "what would the atmosphere side have to supply, hour by hour, to close
    the LST gap?" -- the atmosphere-side ceiling complementary to the
    parameter-side ceiling, and the diurnal shape any new closure must
    reproduce.

``slab``
    One-parameter finite-capacity surface-layer closure (no table scaling).
    A slab air-temperature anomaly dT relative to the reanalysis air
    temperature relaxes under the surface sensible heat flux H (downward
    positive, W m-2):
        C_s d(dT)/dt = -H - C_s/tau_s * dT,   C_s = rho*cp*h_s,
    discretized explicitly per 30-min step as
        dT_{n+1} = beta*dT_n + (1-beta)*(-lambda_s*H_n),
        beta = exp(-dt/tau_s),  tau_s = lambda_s*rho*cp*h_s,
    so the steady state is dT = -lambda_s*H with lambda_s = tau_s/C_s.
    h_s is fixed (default 600 m), leaving the single free parameter
    lambda_s in [0, 0.05] K/(W m-2); lambda_s = 0 reproduces the baseline
    exactly.  The relaxation term is not optional: the undamped
    quasi-steady form dT = -lambda_s*H is a positive feedback under
    nocturnal stable stratification (cooler air -> larger downward H ->
    cooler air) and diverges discretely once lambda_s*rho*cp*C_H*U
    approaches unity, which the first version of this experiment
    demonstrated numerically (NaN at lambda_s ~ 0.03).

Day convention, windows, loss, and evaluation mirror
modis_calibration_heldout.py exactly, so the numbers are directly
comparable with the parameter-scale experiments (v4/v5).
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

TAIR_BOUND_K = 8.0
WIND_SW_BOUND = 2.0
LAMBDA_BOUND = 0.05
RHO_CP = 1.2 * 1005.7  # J m-3 K-1, reference near-surface air heat capacity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("attribution", "slab"), required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--state-archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-days", default="271,281,291,301,311,321,331,341,351,361")
    parser.add_argument("--validation-days", default="91,101,111,121,131,141,151,161,171,181")
    parser.add_argument("--column-stride", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Adam lr for the leading group (dtair K or lambda); "
                             "log-scales always use 0.02")
    parser.add_argument("--init-lambda", type=float, default=0.001,
                        help="slab mode: initial lambda_s K/(W m-2); must be > 0 "
                             "so the recurrence is active and differentiable")
    parser.add_argument("--slab-depth", type=float, default=600.0,
                        help="slab mode: fixed effective depth h_s (m) setting "
                             "tau_s = lambda_s*rho*cp*h_s")
    parser.add_argument("--lst-error-k", type=float, default=1.0)
    parser.add_argument("--probe", action="store_true",
                        help="one-day finite-difference leverage check per control, then exit")
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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    if args.learning_rate is None:
        args.learning_rate = 0.15 if args.mode == "attribution" else 0.004

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
    # Structural control vector (no parameter tables involved).
    # ------------------------------------------------------------------
    if args.mode == "attribution":
        dtair = torch.nn.Parameter(torch.zeros(8, dtype=torch.float64, device=args.device))
        wind_log = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device=args.device))
        sw_log = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device=args.device))
        parameters = [{"params": [dtair], "lr": args.learning_rate},
                      {"params": [wind_log, sw_log], "lr": 0.02}]

        def control_names() -> list[str]:
            return [f"dtair_h{3*b:02d}_{3*b+3:02d}" for b in range(8)] + ["wind_scale", "sw_scale"]

        def control_values() -> list[float]:
            return ([float(v) for v in dtair.detach().cpu()]
                    + [float(wind_log.exp()), float(sw_log.exp())])

        def set_probe(index: int, value: float) -> None:
            with torch.no_grad():
                dtair.zero_(); wind_log.zero_(); sw_log.zero_()
                if index < 8:
                    dtair[index] = value
                elif index == 8:
                    wind_log.fill_(value)
                else:
                    sw_log.fill_(value)
    else:
        lam = torch.nn.Parameter(
            torch.tensor(args.init_lambda, dtype=torch.float64, device=args.device))

        def control_names() -> list[str]:
            return ["lambda_s"]

        def control_values() -> list[float]:
            return [float(lam.detach())]

        def set_probe(index: int, value: float) -> None:
            with torch.no_grad():
                lam.fill_(value)

    optimizer = torch.optim.Adam(parameters) if args.mode == "attribution" \
        else torch.optim.Adam([lam], lr=args.learning_rate)

    all_parameters = [p for group in parameters for p in group["params"]] if args.mode == "attribution" else [lam]

    def clamp_parameters() -> None:
        with torch.no_grad():
            if args.mode == "attribution":
                dtair.clamp_(-TAIR_BOUND_K, TAIR_BOUND_K)
                wind_log.clamp_(-math.log(WIND_SW_BOUND), math.log(WIND_SW_BOUND))
                sw_log.clamp_(-math.log(WIND_SW_BOUND), math.log(WIND_SW_BOUND))
            else:
                lam.clamp_(0.0, LAMBDA_BOUND)

    forcing_options = SimpleNamespace(
        shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
        shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
        stress_u=None, stress_v=None, vup=None,
    )

    def make_advance(step_static_base, column_lat, column_lon, forcing_arrays):
        lon_hours = torch.as_tensor(np.asarray(column_lon, dtype=np.float64) / 15.0,
                                    dtype=torch.float64, device=args.device)
        cache = {"heat": None, "dt_air": None}  # previous-step H [W m-2], slab anomaly [K]

        def advance(current_state, physical_step: int):
            fraction = model._tile_frac(step_static_base, config.n_tile, state=current_state)
            runtime = interpolated_runtime(
                forcing_arrays, physical_step, DT_SECONDS, column_lat, column_lon, args.device,
            )
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
            if args.mode == "attribution":
                utc_hour = (physical_step % STEPS_PER_DAY) * DT_SECONDS / 3600.0
                local_bin = torch.floor(((utc_hour + lon_hours) % 24.0) / 3.0).long()
                forcing = replace(
                    forcing,
                    t=forcing.t + dtair[local_bin],
                    u=forcing.u * wind_log.exp(),
                    v=forcing.v * wind_log.exp(),
                    fr_so=forcing.fr_so * sw_log.exp(),
                )
            else:
                if cache["dt_air"] is not None:
                    forcing = replace(forcing, t=forcing.t + cache["dt_air"])
            new_state = model(current_state, forcing, step_static)[0]
            with torch.no_grad():
                cache["heat"] = (fraction * model._last_pahfsti).sum(dim=1).detach().clone()
            if args.mode == "slab":
                # Keep the recurrence in the autograd graph: under an outer
                # no_grad context these ops degrade to constants, inside the
                # AD window they chain d(loss)/d(lambda) through every step.
                if float(lam.detach()) > 1e-9:
                    tau = lam * RHO_CP * args.slab_depth
                    beta = torch.exp(torch.as_tensor(-DT_SECONDS, dtype=torch.float64, device=args.device) / tau)
                    previous = cache["dt_air"] if cache["dt_air"] is not None \
                        else torch.zeros_like(cache["heat"])
                    cache["dt_air"] = beta * previous + (1.0 - beta) * (-lam * cache["heat"])
                else:
                    cache["dt_air"] = None
            return new_state

        def reset_cache():
            cache["heat"] = None
            cache["dt_air"] = None

        return advance, reset_cache

    advance_subset, reset_subset = make_advance(static, lat, lon, forcing_subset)
    advance_full, reset_full = make_advance(static_full, full_lat, full_lon, forcing_full)

    def day_predictions(day: int, with_gradient: bool, advance, reset, base, static_base, width: int, observed_day: dict):
        """Return view-time LST predictions and their target steps for obs day D."""
        archive_step = (day - 1) * STEPS_PER_DAY
        matching = np.flatnonzero(archive_steps == archive_step)
        if matching.size != 1:
            raise ValueError(f"state archive has no unique record for step {archive_step}")
        state = restore_state(copy.deepcopy(base), static_base, model, archive, int(matching[0]),
                              slice(None, None, args.column_stride) if width == column_count else slice(None),
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
                    mask = torch.as_tensor(period_steps[period] == completed, dtype=torch.bool, device=args.device)
                    predictions[period] = torch.where(mask, state.radiative_skin_t, predictions[period])
        return predictions, period_steps

    area_weight = np.cos(np.deg2rad(lat))

    def day_loss_and_stats(predictions, observed_day, period_steps):
        """Weighted loss (tensor) and per-period bias/RMSE for one day."""
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
        loss = torch.stack(loss_terms).sum() if loss_terms else torch.zeros((), dtype=torch.float64, device=args.device)
        return loss, stats

    def subset_day_loss(day: int, observed_day: dict):
        predictions, period_steps = day_predictions(
            day, False, advance_subset, reset_subset, base_state, static, column_count, observed_day,
        )
        loss, _ = day_loss_and_stats(predictions, observed_day, period_steps)
        return float(loss)

    # ------------------------------------------------------------------
    # Optional leverage probe: dLoss per control on one day.
    # ------------------------------------------------------------------
    if args.probe:
        day = calibration_days[0]
        observed_day = single_day(observations_calibration, 0)
        set_probe(0, 0.0)
        reference = subset_day_loss(day, observed_day)
        print(json.dumps({"probe_day": day, "reference_loss": round(reference, 4)}), flush=True)
        probe_log = []
        for index in range(len(control_names())):
            deltas = []
            if args.mode == "attribution":
                values = [0.5 if index < 8 else 0.1, -0.5 if index < 8 else -0.1]
            else:
                values = [0.004, 0.002]  # one-sided: negative lambda is unphysical
            for value in values:
                set_probe(index, value)
                deltas.append(subset_day_loss(day, observed_day) - reference)
            set_probe(0, 0.0)
            entry = {"control": control_names()[index],
                     "delta_plus": round(deltas[0], 4), "delta_minus": round(deltas[1], 4),
                     "abs_mean_delta": round((abs(deltas[0]) + abs(deltas[1])) / 2.0, 4)}
            probe_log.append(entry)
            print(json.dumps(entry), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "structural_probe.json").write_text(
            json.dumps({"mode": args.mode, "probe_day": day, "reference_loss": reference,
                        "controls": probe_log}, indent=1) + "\n")
        print(json.dumps({"probe": "done"}), flush=True)
        return

    # ------------------------------------------------------------------
    # Adam optimization on the autumn-winter window (day+night loss).
    # ------------------------------------------------------------------
    history = []
    started = time.perf_counter()
    for iteration in range(args.iterations + 1):
        training = iteration > 0
        day_stats = {}
        total_loss = 0.0
        for group_day, day in enumerate(calibration_days):
            observed_day = single_day(observations_calibration, group_day)
            predictions, period_steps = day_predictions(
                day, training, advance_subset, reset_subset, base_state, static, column_count, observed_day,
            )
            loss_day, stats = day_loss_and_stats(predictions, observed_day, period_steps)
            if training:
                loss_day.backward()
            total_loss += float(loss_day.detach())
            day_stats[str(day)] = stats
        if training:
            optimizer.step()
            clamp_parameters()
            for parameter in all_parameters:
                if parameter.grad is not None:
                    parameter.grad = None
        record = {
            "iteration": iteration,
            "loss": total_loss,
            "controls": control_values(),
            "days": day_stats,
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "structural_history.json").write_text(json.dumps(history, indent=1) + "\n")
        aggregate = {}
        for period in PERIODS:
            entries = [d[period] for d in day_stats.values() if d[period]["rmse_K"] is not None]
            aggregate[period] = {
                "rmse_K": float(np.sqrt(np.mean([e["rmse_K"] ** 2 for e in entries]))) if entries else None,
                "bias_K": float(np.mean([e["bias_K"] for e in entries])) if entries else None,
            }
        print(json.dumps({"iteration": iteration, "loss": round(total_loss, 4),
                          "controls": [round(v, 4) for v in record["controls"]],
                          "subset_rmse": {k: (round(v["rmse_K"], 3) if v["rmse_K"] is not None else None)
                                          for k, v in aggregate.items()},
                          "elapsed_s": round(record["elapsed_s"], 1)}), flush=True)

    # ------------------------------------------------------------------
    # Post-optimization evaluation: prior (neutral) vs optimized, both
    # windows, full domain.
    # ------------------------------------------------------------------
    def evaluate_window(days: list[int], observed: dict, label: str) -> dict:
        accum = {period: {"weighted_resid": 0.0, "sq": 0.0, "weight_sum": 0.0, "count": 0} for period in PERIODS}
        for group_day, day in enumerate(days):
            observed_day = single_day(observed, group_day)
            predictions, period_steps = day_predictions(
                day, False, advance_full, reset_full, base_state_full, static_full, nvalid, observed_day,
            )
            for period in PERIODS:
                observation = observed_day[period]["lst"][0]
                valid = observed_day[period]["strict"][0] & np.isfinite(observation) & (period_steps[period] >= 1)
                if not valid.any():
                    continue
                residual = predictions[period][torch.as_tensor(valid, device=args.device)].cpu().numpy() - observation[valid]
                weights = np.cos(np.deg2rad(full_lat))[valid]
                accum[period]["weighted_resid"] += float((residual * weights).sum())
                accum[period]["sq"] += float((residual ** 2 * weights).sum())
                accum[period]["weight_sum"] += float(weights.sum())
                accum[period]["count"] += int(residual.size)
        summary = {}
        for period, entry in accum.items():
            if entry["count"] == 0:
                summary[period] = {"n": 0, "bias_K": None, "rmse_K": None}
                continue
            summary[period] = {
                "n": entry["count"],
                "bias_K": entry["weighted_resid"] / entry["weight_sum"],
                "rmse_K": math.sqrt(entry["sq"] / entry["weight_sum"]),
            }
        print(json.dumps({"eval": label, "summary": summary}), flush=True)
        return summary

    optimized_controls = control_values()

    def set_neutral():
        if args.mode == "attribution":
            with torch.no_grad():
                dtair.zero_(); wind_log.zero_(); sw_log.zero_()
        else:
            with torch.no_grad():
                lam.fill_(0.0)

    def set_optimized():
        if args.mode == "attribution":
            with torch.no_grad():
                dtair.copy_(torch.as_tensor(optimized_controls[:8], dtype=torch.float64, device=args.device))
                wind_log.fill_(math.log(max(optimized_controls[8], 1e-6)))
                sw_log.fill_(math.log(max(optimized_controls[9], 1e-6)))
        else:
            with torch.no_grad():
                lam.fill_(optimized_controls[0])

    set_neutral()
    baseline_full = {
        "calibration_window": evaluate_window(calibration_days, full_calibration_obs, "baseline_calibration"),
        "validation_window": evaluate_window(validation_days, full_validation_obs, "baseline_validation"),
    }
    set_optimized()
    optimized_full = {
        "calibration_window": evaluate_window(calibration_days, full_calibration_obs, "optimized_calibration"),
        "validation_window": evaluate_window(validation_days, full_validation_obs, "optimized_validation"),
    }

    final = {
        "config": {
            "mode": args.mode,
            "control_names": control_names(),
            "slab_depth_m": args.slab_depth if args.mode == "slab" else None,
            "init_lambda": args.init_lambda if args.mode == "slab" else None,
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
        "final_controls": optimized_controls,
        "baseline_full_domain": baseline_full,
        "optimized_full_domain": optimized_full,
    }
    (args.output.parent / "structural_result.json").write_text(json.dumps(final, indent=1) + "\n")

    # ------------------------------------------------------------------
    # Figure.
    # ------------------------------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    names = control_names()
    iterations_axis = [h["iteration"] for h in history]
    if args.mode == "attribution":
        centers = np.arange(8) * 3.0 + 1.5
        axes[0].bar(centers, optimized_controls[:8], width=2.6, color="tab:blue")
        axes[0].axhline(0.0, color="gray", linewidth=0.8)
        axes[0].set(xlabel="local hour", ylabel="delta T_air (K)",
                    title=f"Optimal pseudo-forcing profile\nwind x{optimized_controls[8]:.3f}, SW x{optimized_controls[9]:.3f}")
        axes[0].grid(alpha=0.3)
    else:
        axes[0].plot(iterations_axis, [h["controls"][0] for h in history], marker=".")
        axes[0].set(xlabel="iteration", ylabel="lambda_s  K/(W m-2)",
                    title="Slab coupling coefficient")
        axes[0].grid(alpha=0.3)
    for period, color in zip(PERIODS, ("tab:red", "tab:blue")):
        values = []
        for h in history:
            entries = [d[period]["rmse_K"] for d in h["days"].values() if d[period]["rmse_K"] is not None]
            values.append(float(np.sqrt(np.mean(np.square(entries)))) if entries else float("nan"))
        axes[1].plot(iterations_axis, values, color=color, label=period)
    axes[1].set(xlabel="iteration", ylabel="LST RMSE (K)", title="Calibration window (training subset)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    labels, positions, heights, colors = [], [], [], []
    for window_index, window in enumerate(("calibration_window", "validation_window")):
        for period_index, period in enumerate(PERIODS):
            for config_index, (config_name, payload) in enumerate(
                (("baseline", baseline_full), ("optimized", optimized_full))
            ):
                value = payload[window][period]["rmse_K"]
                labels.append(f"{window.split('_')[0][:3]} {period}\n{config_name}")
                positions.append(window_index * 2.6 + period_index * 1.3 + config_index * 0.6)
                heights.append(value if value is not None else 0.0)
                colors.append(f"C{config_index}")
    axes[2].bar(positions, heights, color=colors, width=0.55)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels, fontsize=6, rotation=30, ha="right")
    axes[2].set(ylabel="LST RMSE (K)", title="RMSE: prior vs structural closure")
    axes[2].grid(alpha=0.3, axis="y")
    figure.suptitle(f"Structural closure experiment ({args.mode}) against MODIS LST", fontsize=11)
    figure.savefig(args.output.parent / "structural_closure.png", dpi=260)
    plt.close(figure)
    print(json.dumps({"done": str(args.output.parent / "structural_result.json")}), flush=True)


if __name__ == "__main__":
    main()
