#!/usr/bin/env python3
"""Real-observation differentiable calibration with a held-out seasonal window.

Calibrates logarithmic scaling controls against strict-QC MODIS Terra LST in
the well-conditioned autumn--winter window (information condition number ~1.2e2
from ``modis_parameter_identifiability.py``), then evaluates prior and
calibrated parameters in the held-out, near-degenerate spring--summer window
(condition number ~5.5e7).

Two control vectors are supported:

``uniform3``
    Three uniform scales of the RVLAI/RVRSMIN/RVZ0M vegetation tables (the
    identifiability-experiment directions).  Retained as the ablation
    baseline: these directions absorb only ~3% of the LST misfit.

``extended`` (default)
    Eight controls with direct skin-temperature leverage: the same three
    vegetation tables split into high- and low-vegetation class masks
    (six controls), a joint scaling of the dry and saturated soil thermal
    conductivity end-members (``lambdadry`` + ``lamsat1``, one control), and
    the soil heat-capacity table (``rcsoil``, one control).

Day convention (identical to ``modis_parameter_identifiability.py``): for a
zero-based observation day index D, the archived state at step (D-1)*48 is
restored, day D advances 48 steps without gradients (parameter adjustment),
and day D+1 runs 48 steps sampled at the cell-specific Terra view times of
obs[D].
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--scale-bound", type=float, default=2.0)
    parser.add_argument("--lst-error-k", type=float, default=1.0)
    parser.add_argument("--controls", choices=("uniform3", "extended"), default="extended")
    parser.add_argument("--probe", action="store_true",
                        help="one-day finite-difference leverage check per control, then exit")
    parser.add_argument("--twin", default="",
                        help="comma list control:scale for an observing-system simulation "
                             "experiment, e.g. rvz0m_low:0.7,rcsoil:1.4 (synthetic MODIS obs)")
    parser.add_argument("--twin-noise", type=float, default=1.0,
                        help="Gaussian observation noise standard deviation (K) in twin mode")
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
        learnable_soil=("lambdadry", "lamsat1", "rcsoil") if args.controls == "extended" else (),
    )
    model = LandSurface(config).to(args.device)

    # ------------------------------------------------------------------
    # Control vector: multiplicative log-scales over (table, mask) blocks.
    # ------------------------------------------------------------------
    class Control:
        def __init__(self, name, entries):
            self.name = name
            self.entries = entries  # list of (table_tensor, base_tensor, mask_tensor)

    def class_mask(classes, length):
        mask = torch.zeros(length, dtype=torch.bool, device=args.device)
        for index in classes:
            mask[int(index)] = True
        return mask

    controls = []
    if args.controls == "uniform3":
        for field in PARAMETER_NAMES:
            table = getattr(model.veg_table, field)
            controls.append(Control(field, [(table, table.detach().clone(),
                                             torch.ones_like(table, dtype=torch.bool))]))
    else:
        for static_field, label in (("veg_type_high", "high"), ("veg_type_low", "low")):
            source = getattr(static_full, static_field, None)
            if source is None:
                raise ValueError(f"extended controls require {static_field} in the static state")
            classes = [int(c) for c in torch.unique(source).tolist() if int(c) > 0]
            print(json.dumps({"mask": static_field, "classes": classes}), flush=True)
            for field in PARAMETER_NAMES:
                table = getattr(model.veg_table, field)
                mask = class_mask(classes, table.shape[0])
                controls.append(Control(f"{field}_{label}",
                                        [(table, table.detach().clone(), mask)]))
        soil_classes = [int(c) for c in torch.unique(static_full.soil_type).tolist() if int(c) > 0]
        print(json.dumps({"mask": "soil_type", "classes": soil_classes}), flush=True)
        soil_mask = class_mask(soil_classes, model.soil_table.lambdadry.shape[0])
        controls.append(Control("lambda_soil", [
            (model.soil_table.lambdadry, model.soil_table.lambdadry.detach().clone(), soil_mask),
            (model.soil_table.lamsat1, model.soil_table.lamsat1.detach().clone(), soil_mask),
        ]))
        controls.append(Control("rcsoil", [
            (model.soil_table.rcsoil, model.soil_table.rcsoil.detach().clone(), soil_mask),
        ]))
    control_names = [control.name for control in controls]

    def apply_scales(scales: torch.Tensor) -> None:
        with torch.no_grad():
            for control, scale in zip(controls, scales):
                for table, base, mask in control.entries:
                    table.copy_(torch.where(mask, base * scale.exp(), base))

    forcing_options = SimpleNamespace(
        shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
        shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
        stress_u=None, stress_v=None, vup=None,
    )

    def make_advance(step_static_base, column_lat, column_lon, forcing_arrays):
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
            return model(current_state, forcing, step_static)[0]
        return advance

    advance_subset = make_advance(static, lat, lon, forcing_subset)
    advance_full = make_advance(static_full, full_lat, full_lon, forcing_full)

    def day_predictions(day: int, with_gradient: bool, advance, base, static_base, width: int, observed_day: dict):
        """Return view-time LST predictions and their target steps for obs day D."""
        archive_step = (day - 1) * STEPS_PER_DAY
        matching = np.flatnonzero(archive_steps == archive_step)
        if matching.size != 1:
            raise ValueError(f"state archive has no unique record for step {archive_step}")
        state = restore_state(copy.deepcopy(base), static_base, model, archive, int(matching[0]),
                              slice(None, None, args.column_stride) if width == column_count else slice(None),
                              args.device)
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
            day, False, advance_subset, base_state, static, column_count, observed_day,
        )
        loss, _ = day_loss_and_stats(predictions, observed_day, period_steps)
        return float(loss)

    zero_scales = torch.zeros(len(controls), dtype=torch.float64, device=args.device)
    apply_scales(zero_scales)

    # ------------------------------------------------------------------
    # Twin (OSSE) mode: synthetic MODIS observations from a known-truth
    # parameter perturbation, same operator, masks, and view times.
    # ------------------------------------------------------------------
    truth_scales = None
    if args.twin:
        truth_scales = zero_scales.clone()
        requested = [item.split(":") for item in args.twin.split(",") if item.strip()]
        for name, value in requested:
            if name not in control_names:
                raise ValueError(f"twin control {name!r} is not in the control vector")
            truth_scales[control_names.index(name)] = math.log(float(value))
        apply_scales(truth_scales)

        def build_synthetic(days: list[int], observed_real: dict, seed_offset: int) -> dict:
            synthetic = {
                period: {"lst": np.full_like(observed_real[period]["lst"], np.nan),
                         "time": observed_real[period]["time"],
                         "strict": observed_real[period]["strict"]}
                for period in PERIODS
            }
            generator = torch.Generator()
            generator.manual_seed(args.seed + seed_offset)
            for group_day, day in enumerate(days):
                observed_day = single_day(observed_real, group_day)
                predictions, period_steps = day_predictions(
                    day, False, advance_subset, base_state, static, column_count, observed_day,
                )
                for period in PERIODS:
                    valid = (observed_real[period]["strict"][group_day]
                             & np.isfinite(observed_real[period]["lst"][group_day])
                             & (period_steps[period] >= 1))
                    if not valid.any():
                        continue
                    truth_values = predictions[period][torch.as_tensor(valid, device=args.device)].detach().cpu().numpy()
                    noise = torch.randn(int(valid.sum()), generator=generator).numpy() * args.twin_noise
                    synthetic[period]["lst"][group_day][valid] = truth_values + noise
            return synthetic

        synthetic_calibration = build_synthetic(calibration_days, observations_calibration, 0)
        synthetic_validation = build_synthetic(validation_days, observations_validation, 1000)
        apply_scales(zero_scales)
        observations_calibration = synthetic_calibration
        observations_validation = synthetic_validation
        full_calibration_obs = None
        full_validation_obs = None
        print(json.dumps({"twin": "observations synthesized",
                          "noise_k": args.twin_noise,
                          "truth_scales": [round(v, 4) for v in truth_scales.exp().cpu().tolist()]}), flush=True)

    # ------------------------------------------------------------------
    # Optional leverage probe: dLoss per control at +/-10% scale on one day.
    # ------------------------------------------------------------------
    if args.probe:
        day = calibration_days[0]
        observed_day = single_day(observations_calibration, 0)
        reference = subset_day_loss(day, observed_day)
        print(json.dumps({"probe_day": day, "reference_loss": round(reference, 4)}), flush=True)
        probe_log = []
        for index, control in enumerate(controls):
            deltas = []
            for sign in (+1.0, -1.0):
                scales = zero_scales.clone()
                scales[index] = sign * 0.1
                apply_scales(scales)
                deltas.append(subset_day_loss(day, observed_day) - reference)
            apply_scales(zero_scales)
            entry = {"control": control.name,
                     "loss_plus10": round(deltas[0], 4), "loss_minus10": round(deltas[1], 4),
                     "abs_mean_delta": round((abs(deltas[0]) + abs(deltas[1])) / 2.0, 4)}
            probe_log.append(entry)
            print(json.dumps(entry), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "modis_calibration_probe.json").write_text(
            json.dumps({"probe_day": day, "reference_loss": reference, "controls": probe_log}, indent=1) + "\n")
        print(json.dumps({"probe": "done"}), flush=True)
        return

    # ------------------------------------------------------------------
    # Adam calibration on the well-conditioned autumn-winter window.
    # ------------------------------------------------------------------
    log_scales = torch.zeros(len(controls), dtype=torch.float64, device=args.device, requires_grad=True)
    optimizer = torch.optim.Adam([log_scales], lr=args.learning_rate)
    bound = math.log(args.scale_bound)
    history = []
    started = time.perf_counter()
    for iteration in range(args.iterations + 1):
        training = iteration > 0
        day_stats = {}
        total_loss = 0.0
        for group_day, day in enumerate(calibration_days):
            observed_day = single_day(observations_calibration, group_day)
            predictions, period_steps = day_predictions(
                day, training, advance_subset, base_state, static, column_count, observed_day,
            )
            loss_day, stats = day_loss_and_stats(predictions, observed_day, period_steps)
            if training:
                loss_day.backward()
            total_loss += float(loss_day.detach())
            day_stats[str(day)] = stats
        if training:
            projected = []
            for control, scale in zip(controls, log_scales):
                total = None
                for table, base, mask in control.entries:
                    if table.grad is None:
                        continue
                    term = (table.grad * base * mask * scale.exp()).sum()
                    total = term if total is None else total + term
                projected.append(total if total is not None else
                                 torch.zeros((), dtype=torch.float64, device=args.device))
            log_scales.grad = torch.stack(projected)
            optimizer.step()
            with torch.no_grad():
                log_scales.clamp_(-bound, bound)
            apply_scales(log_scales.detach())
            for control in controls:
                for table, _, _ in control.entries:
                    if table.grad is not None:
                        table.grad = None
        record = {
            "iteration": iteration,
            "loss": total_loss,
            "scales": [float(v) for v in log_scales.detach().exp().cpu()],
            "days": day_stats,
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "modis_calibration_history.json").write_text(json.dumps(history, indent=1) + "\n")
        aggregate = {}
        for period in PERIODS:
            entries = [d[period] for d in day_stats.values() if d[period]["rmse_K"] is not None]
            aggregate[period] = {
                "rmse_K": float(np.sqrt(np.mean([e["rmse_K"] ** 2 for e in entries]))) if entries else None,
                "bias_K": float(np.mean([e["bias_K"] for e in entries])) if entries else None,
            }
        print(json.dumps({"iteration": iteration, "loss": round(total_loss, 4),
                          "scales": [round(v, 4) for v in record["scales"]],
                          "subset_rmse": {k: (round(v["rmse_K"], 3) if v["rmse_K"] is not None else None)
                                          for k, v in aggregate.items()},
                          "elapsed_s": round(record["elapsed_s"], 1)}), flush=True)

    # ------------------------------------------------------------------
    # Post-optimization evaluation: prior vs calibrated in both windows.
    # Real observations are evaluated on the full domain; the twin
    # experiment is evaluated on the training-column subset.
    # ------------------------------------------------------------------
    if truth_scales is None:
        eval_advance, eval_base, eval_static, eval_width, eval_area = (
            advance_full, base_state_full, static_full, nvalid, np.cos(np.deg2rad(full_lat)),
        )
        observed_calibration_eval = full_calibration_obs
        observed_validation_eval = full_validation_obs
    else:
        eval_advance, eval_base, eval_static, eval_width, eval_area = (
            advance_subset, base_state, static, column_count, area_weight,
        )
        observed_calibration_eval = observations_calibration
        observed_validation_eval = observations_validation

    def evaluate_window(days: list[int], observed: dict, label: str) -> dict:
        accum = {period: {"weighted_resid": 0.0, "sq": 0.0, "weight_sum": 0.0, "count": 0} for period in PERIODS}
        for group_day, day in enumerate(days):
            observed_day = single_day(observed, group_day)
            predictions, period_steps = day_predictions(
                day, False, eval_advance, eval_base, eval_static, eval_width, observed_day,
            )
            for period in PERIODS:
                observation = observed_day[period]["lst"][0]
                valid = observed_day[period]["strict"][0] & np.isfinite(observation) & (period_steps[period] >= 1)
                if not valid.any():
                    continue
                residual = predictions[period][torch.as_tensor(valid, device=args.device)].cpu().numpy() - observation[valid]
                weights = eval_area[valid]
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

    apply_scales(zero_scales)
    baseline_full = {
        "calibration_window": evaluate_window(calibration_days, observed_calibration_eval, "baseline_calibration"),
        "validation_window": evaluate_window(validation_days, observed_validation_eval, "baseline_validation"),
    }
    apply_scales(log_scales.detach())
    calibrated_full = {
        "calibration_window": evaluate_window(calibration_days, observed_calibration_eval, "calibrated_calibration"),
        "validation_window": evaluate_window(validation_days, observed_validation_eval, "calibrated_validation"),
    }

    recovery = None
    if truth_scales is not None:
        final_values = log_scales.detach().exp().cpu()
        truth_values = truth_scales.exp().cpu()
        recovery = [
            {"control": name, "truth": float(truth), "recovered": float(value),
             "error_pct": 100.0 * abs(float(value) - float(truth)) / float(truth)}
            for name, truth, value in zip(control_names, truth_values, final_values)
            if abs(float(truth) - 1.0) > 1e-12
        ]
        print(json.dumps({"recovery": recovery}), flush=True)

    final = {
        "config": {
            "controls_mode": args.controls,
            "control_names": control_names,
            "twin": args.twin if args.twin else None,
            "twin_noise_k": args.twin_noise if args.twin else None,
            "calibration_days": calibration_days,
            "validation_days": validation_days,
            "column_stride": args.column_stride,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "scale_bound": args.scale_bound,
            "lst_error_k": args.lst_error_k,
            "seed": args.seed,
            "case": str(args.case),
            "data": str(args.data),
            "observations": str(args.observations),
            "state_archive": str(args.state_archive),
        },
        "final_scales": [float(v) for v in log_scales.detach().exp().cpu()],
        "recovery": recovery,
        "baseline_full_domain": baseline_full,
        "calibrated_full_domain": calibrated_full,
    }
    (args.output.parent / "modis_calibration_result.json").write_text(json.dumps(final, indent=1) + "\n")

    # Figure: scale history, calibration-window RMSE, and window comparison.
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    iterations = [h["iteration"] for h in history]
    for index, name in enumerate(control_names):
        axes[0].plot(iterations, [h["scales"][index] for h in history], marker=".", label=name)
        if truth_scales is not None and abs(float(truth_scales[index].exp()) - 1.0) > 1e-12:
            axes[0].axhline(float(truth_scales[index].exp()), linestyle="--", linewidth=1.0, color=f"C{index}")
    axes[0].axhline(1.0, color="gray", linewidth=0.8)
    axes[0].set(xlabel="iteration", ylabel="multiplicative scale", title="Parameter-table scales")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(alpha=0.3)

    for period, color in zip(PERIODS, ("tab:red", "tab:blue")):
        values = []
        for h in history:
            entries = [d[period]["rmse_K"] for d in h["days"].values() if d[period]["rmse_K"] is not None]
            values.append(float(np.sqrt(np.mean(np.square(entries)))) if entries else float("nan"))
        axes[1].plot(iterations, values, color=color, label=period)
    axes[1].set(xlabel="iteration", ylabel="LST RMSE (K)", title="Calibration window (training subset)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    labels, positions, heights, colors = [], [], [], []
    for window_index, window in enumerate(("calibration_window", "validation_window")):
        for period_index, period in enumerate(PERIODS):
            for config_index, (config_name, payload) in enumerate(
                (("baseline", baseline_full), ("calibrated", calibrated_full))
            ):
                value = payload[window][period]["rmse_K"]
                labels.append(f"{window.split('_')[0][:3]} {period}\n{config_name}")
                positions.append(window_index * 2.6 + period_index * 1.3 + config_index * 0.6)
                heights.append(value if value is not None else 0.0)
                colors.append(f"C{config_index}")
    axes[2].bar(positions, heights, color=colors, width=0.55)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels, fontsize=6, rotation=30, ha="right")
    axes[2].set(ylabel="LST RMSE (K)", title="RMSE: prior vs calibrated")
    axes[2].grid(alpha=0.3, axis="y")
    figure.suptitle("Differentiable calibration against MODIS LST: conditioned-window training, held-out evaluation", fontsize=11)
    figure.savefig(args.output.parent / "modis_calibration_heldout.png", dpi=260)
    plt.close(figure)
    print(json.dumps({"done": str(args.output.parent / "modis_calibration_result.json")}), flush=True)


if __name__ == "__main__":
    main()
