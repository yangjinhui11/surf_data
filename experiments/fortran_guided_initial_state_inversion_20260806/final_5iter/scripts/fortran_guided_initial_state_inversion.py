#!/usr/bin/env python3
"""Fortran-guided initial-state inversion for autonomous offline SURF.

This experiment is deliberately distinct from a synthetic-twin calibration:
the targets are saved from a separately executed Fortran offline integration.
One representative warm, snow-free CMFD column is selected from the Fortran
day-10 and day-20 states alone.  A prescribed error is added to its initial
top-layer soil-water mass.  Reverse-mode derivatives of the PyTorch
day-10 state construct damped Gauss--Newton updates, while the day-20 Fortran
state is held out from the update calculation.

It is a compact prototype for gradient-based land-state estimation, not an
observational data-assimilation experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from netCDF4 import Dataset


FORCING_NAMES = (
    "tair_K",
    "qair_kg_kg",
    "psurf_Pa",
    "swdown_W_m2",
    "lwdown_W_m2",
    "wind_m_s",
    "precip_kg_m2_s",
)
CONTROL_FIELDS = (
    "soil_water_layer_1",
    "soil_water_layer_2",
    "soil_temperature_layer_1",
)
CONTROL_SCALES = np.asarray((5.0, 10.0, 1.0), dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="Torch annual-run case with restartin.nc and metadata.json")
    parser.add_argument("--data", type=Path, required=True, help="Processed CMFD valid-column dataset")
    parser.add_argument("--reference", type=Path, required=True, help="Fortran 10-day-output o_gg.nc")
    parser.add_argument("--source", type=Path, required=True, help="SURF PyTorch source directory")
    parser.add_argument("--runner", type=Path, required=True, help="Directory containing the CMFD runner helpers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target-step", type=int, default=480, help="Fortran step used by the update objective")
    parser.add_argument("--validation-step", type=int, default=960, help="Fortran step kept out of the update objective")
    parser.add_argument("--initial-bias-kg-m2", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--damping", type=float, default=1.0e-3)
    parser.add_argument("--maximum-update-kg-m2", type=float, default=3.0)
    parser.add_argument("--maximum-bias-kg-m2", type=float, default=10.0)
    parser.add_argument("--column-index", type=int, default=None, help="Override the deterministic representative-column selection")
    parser.add_argument("--render-existing", type=Path, default=None, help="Redraw an existing result summary without rerunning SURF")
    return parser.parse_args()


def record_index(dataset: Dataset, step: int) -> int:
    records = np.flatnonzero(np.asarray(dataset["timestp"][:], dtype=np.int64) == step)
    if records.size != 1:
        raise ValueError(f"expected one Fortran output record for step {step}, found {records.size}")
    return int(records[0])


def reference_fields(dataset: Dataset, record: int) -> dict[str, np.ndarray]:
    return {
        "soil_w": np.asarray(dataset["SoilMoist"][record, :, 0, :], dtype=np.float64).T,
        "soil_t": np.asarray(dataset["SoilTemp"][record, :, 0, :], dtype=np.float64).T,
        "snow_w": np.asarray(dataset["SWE"][record, 0, :], dtype=np.float64),
    }


def choose_representative_column(day10: dict[str, np.ndarray], day20: dict[str, np.ndarray]) -> int:
    """Select a median warm, snow-free column using only Fortran fields."""
    mask = (
        (day10["snow_w"] < 1.0e-6)
        & (day20["snow_w"] < 1.0e-6)
        & (day10["soil_t"][:, 0] > 278.0)
        & (day20["soil_t"][:, 0] > 278.0)
        & (day10["soil_w"][:, 0] > 5.0)
        & (day10["soil_w"][:, 0] < 30.0)
        & (day20["soil_w"][:, 0] > 5.0)
        & (day20["soil_w"][:, 0] < 30.0)
    )
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        raise RuntimeError("no warm, snow-free representative column satisfies the fixed selection rule")
    features = np.stack((
        day10["soil_w"][candidates, 0],
        day10["soil_t"][candidates, 0],
        day20["soil_w"][candidates, 0],
        day20["soil_t"][candidates, 0],
    ), axis=1)
    median = np.median(features, axis=0)
    spread = np.maximum(np.median(np.abs(features - median), axis=0), 1.0e-6)
    return int(candidates[np.argmin(np.sum(((features - median) / spread) ** 2, axis=1))])


def subset_dataclass(value: object, column: int, nvalid: int) -> object:
    selected = slice(column, column + 1)
    updates = {
        field.name: getattr(value, field.name)[selected]
        for field in fields(value)
        if isinstance(getattr(value, field.name), torch.Tensor)
        and getattr(value, field.name).ndim > 0
        and getattr(value, field.name).shape[0] == nvalid
    }
    return replace(value, **updates)


def offline_restart_state(state: object) -> object:
    """Apply the OSM first-call tiled skin-temperature initialization."""
    skin_t = state.radiative_skin_t.unsqueeze(-1).expand_as(state.skin_t).clone()
    skin_t[:, 0] = state.sst
    return replace(state, skin_t=skin_t)


def initial_with_bias(base_state: object, static: object, bias: torch.Tensor) -> object:
    thickness = static.soil_thickness[0]
    mass_to_volume = bias / (1000.0 * thickness)
    soil_w = torch.cat((base_state.soil_w_liq[:, :1] + mass_to_volume, base_state.soil_w_liq[:, 1:]), dim=1)
    return replace(base_state, soil_w_liq=soil_w)


def state_values(state: object, static: object) -> dict[str, torch.Tensor]:
    soil_w = state.soil_w_liq * static.soil_thickness * 1000.0
    values: dict[str, torch.Tensor] = {}
    for level in range(4):
        values[f"soil_water_layer_{level + 1}"] = soil_w[0, level]
        values[f"soil_temperature_layer_{level + 1}"] = state.soil_t[0, level]
    return values


def fortran_values(reference: dict[str, np.ndarray], column: int) -> dict[str, float]:
    values: dict[str, float] = {}
    for level in range(4):
        values[f"soil_water_layer_{level + 1}"] = float(reference["soil_w"][column, level])
        values[f"soil_temperature_layer_{level + 1}"] = float(reference["soil_t"][column, level])
    return values


def as_float_dict(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def run_rollout(
    *,
    bias: torch.Tensor,
    nsteps: int,
    base_state: object,
    static: object,
    forcing_arrays: dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    device: str,
    imported: dict[str, object],
) -> object:
    LandSurface = imported["LandSurface"]
    SurfConfig = imported["SurfConfig"]
    interpolated_runtime = imported["interpolated_runtime"]
    make_forcing = imported["make_forcing"]
    dtype = imported["dtype"]
    config = SurfConfig(
        dtype=dtype,
        device=device,
        differentiable=True,
        validate=False,
        use_farquhar=True,
        use_ags=False,
        levgen=True,
        lelwtl=True,
        lelaiv=False,
        vup_boundary_mode="offline",
    )
    model = LandSurface(config).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = initial_with_bias(base_state, static, bias)
    forcing_options = SimpleNamespace(
        shift_flux=True,
        shift_state=False,
        shift_sw=False,
        shift_lw=False,
        shift_precip=False,
        use_tendencies=True,
        mu0_index="flux",
        lw_mode="net",
        stress_u=None,
        stress_v=None,
        vup=None,
    )
    for step in range(nsteps):
        fraction = model._tile_frac(static, config.n_tile, state=state)
        runtime = interpolated_runtime(
            forcing_arrays, step, 1800.0, lat, lon, device,
        )
        emission = model._surface_emissivity(state, static, fraction, dtype, device)
        step_static = replace(static, emis=emission)
        forcing = make_forcing(
            runtime, 0, state, step_static, forcing_options, tile_frac=fraction,
        )
        precip_amount = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
        forcing = replace(
            forcing,
            rain=precip_amount * (1.0 - runtime.precip_snow_phase),
            snow=precip_amount * runtime.precip_snow_phase,
        )
        state, _ = model(state, forcing, step_static)
    return state


def objective(values: dict[str, torch.Tensor], target: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = torch.stack(tuple(values[name] for name in CONTROL_FIELDS))
    residual = (predicted - target) / scales
    return residual, 0.5 * residual.square().sum()


def evaluation_record(name: str, state: object, static: object, target: dict[str, float]) -> dict[str, object]:
    prediction = as_float_dict(state_values(state, static))
    fields_out = []
    for field in prediction:
        reference = target[field]
        difference = prediction[field] - reference
        fields_out.append({
            "field": field,
            "torch": prediction[field],
            "fortran": reference,
            "difference": difference,
            "absolute_error": abs(difference),
        })
    return {"label": name, "fields": fields_out}


def source_record(source: Path) -> dict[str, object]:
    def command(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=source, text=True).strip()

    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=source)
    return {
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": command(["git", "status", "--short"]).splitlines(),
        "tracked_diff_sha256": __import__("hashlib").sha256(diff).hexdigest(),
    }


def plot_result(
    history: list[dict[str, float | int]],
    validation: list[dict[str, object]],
    output: Path,
    final_bias: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True)
    steps = [int(row["iteration"]) for row in history] + [int(history[-1]["iteration"]) + 1]
    biases = [float(row["bias_before_kg_m2"]) for row in history] + [final_bias]
    axes[0].plot(steps, biases, "o-", color="#1976a2", label="retrieved bias")
    axes[0].axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", label="unperturbed restart")
    axes[0].set(
        xlabel="Gauss-Newton iteration",
        ylabel="Initial top-soil-water bias (kg m$^{-2}$)",
        title="Fortran-guided initial-state retrieval",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fields = CONTROL_FIELDS
    labels = ["soil water L1", "soil water L2", "soil temperature L1"]
    scales = dict(zip(fields, CONTROL_SCALES))
    positions = np.arange(len(fields))
    width = 0.25
    colors = ("#d55e00", "#7570b3", "#1b9e77")
    for offset, evaluation in zip((-width, 0.0, width), validation):
        lookup = {row["field"]: row for row in evaluation["fields"]}
        errors = [lookup[field]["absolute_error"] / scales[field] for field in fields]
        axes[1].bar(positions + offset, errors, width, label=str(evaluation["label"]), color=colors[len(axes[1].patches) // len(fields)])
    axes[1].set(
        xticks=positions,
        xticklabels=labels,
        ylabel="Day-20 absolute error / calibration scale",
        title="Held-out Fortran day-20 comparison",
    )
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.target_step >= args.validation_step:
        raise ValueError("--target-step must precede --validation-step")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.maximum_update_kg_m2 <= 0.0 or args.maximum_bias_kg_m2 <= 0.0:
        raise ValueError("update and bias limits must be positive")
    if args.render_existing is not None:
        existing = json.loads(args.render_existing.read_text())
        args.output.mkdir(parents=True, exist_ok=True)
        plot_result(
            existing["history"],
            existing["held_out_day20"],
            args.output / "fortran_guided_initial_state_inversion.png",
            float(existing["configuration"]["retrieved_bias_kg_m2"]),
        )
        print(json.dumps({"rendered": str(args.output / "fortran_guided_initial_state_inversion.png")}, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SURF_CASE", str(args.case.resolve()))
    os.environ.setdefault("SURF_CODE", str(args.source.resolve()))
    os.environ.setdefault("SURF_DATA", str(args.data.resolve()))
    sys.path[:0] = [str(args.source.resolve()), str(args.runner.resolve())]
    from diagnostic_scripts_compare_realcol import make_forcing
    from replay_full_domain_controlled import DTYPE, load_initial
    from run_full_domain_offline import interpolated_runtime
    from surf_pytorch import LandSurface, SurfConfig

    imported = {
        "LandSurface": LandSurface,
        "SurfConfig": SurfConfig,
        "interpolated_runtime": interpolated_runtime,
        "make_forcing": make_forcing,
        "dtype": DTYPE,
    }
    device = args.device
    with Dataset(args.case / "restartin.nc") as restart:
        nvalid = len(restart.dimensions["lon"])
    lat_all = np.asarray(np.load(args.data / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon_all = np.asarray(np.load(args.data / "lon.npy", mmap_mode="r"), dtype=np.float64)
    if lat_all.size != nvalid or lon_all.size != nvalid:
        raise ValueError("restart and forcing valid-column counts differ")
    with Dataset(args.reference) as reference_file:
        target_record = record_index(reference_file, args.target_step)
        validation_record = record_index(reference_file, args.validation_step)
        target_reference = reference_fields(reference_file, target_record)
        validation_reference = reference_fields(reference_file, validation_record)
    column = args.column_index if args.column_index is not None else choose_representative_column(target_reference, validation_reference)
    if not 0 <= column < nvalid:
        raise ValueError(f"column {column} is outside 0..{nvalid - 1}")
    lat, lon = lat_all[column:column + 1], lon_all[column:column + 1]
    forcing_arrays = {
        name: np.load(args.data / f"{name}.npy", mmap_mode="r")[:, column:column + 1]
        for name in FORCING_NAMES
    }
    loaded_state, loaded_static = load_initial(args.case / "restartin.nc", nvalid, device, lat_all, lon_all)
    base_state = offline_restart_state(subset_dataclass(loaded_state, column, nvalid))
    static = subset_dataclass(loaded_static, column, nvalid)
    del loaded_state, loaded_static
    if device == "cuda":
        torch.cuda.empty_cache()
    target = torch.as_tensor(
        [fortran_values(target_reference, column)[field] for field in CONTROL_FIELDS],
        dtype=DTYPE,
        device=device,
    )
    scales = torch.as_tensor(CONTROL_SCALES, dtype=DTYPE, device=device)
    target_values = fortran_values(target_reference, column)
    validation_values = fortran_values(validation_reference, column)
    base_mass = float((base_state.soil_w_liq[0, 0] * static.soil_thickness[0] * 1000.0).detach().cpu())
    lower_bias = max(-args.maximum_bias_kg_m2, -base_mass + 1.0e-6)
    upper_bias = args.maximum_bias_kg_m2
    bias = torch.tensor(float(np.clip(args.initial_bias_kg_m2, lower_bias, upper_bias)), dtype=DTYPE, device=device, requires_grad=True)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for iteration in range(args.iterations):
        state = run_rollout(
            bias=bias,
            nsteps=args.target_step,
            base_state=base_state,
            static=static,
            forcing_arrays=forcing_arrays,
            lat=lat,
            lon=lon,
            device=device,
            imported=imported,
        )
        residual, loss = objective(state_values(state, static), target, scales)
        jacobian = []
        for component in range(residual.numel()):
            derivative = torch.autograd.grad(
                residual[component], bias, retain_graph=component + 1 < residual.numel(),
            )[0]
            jacobian.append(derivative)
        jacobian_tensor = torch.stack(jacobian)
        gradient = torch.dot(jacobian_tensor, residual.detach())
        curvature = torch.dot(jacobian_tensor, jacobian_tensor)
        raw_update = -gradient / (curvature + args.damping)
        update = torch.clamp(raw_update, -args.maximum_update_kg_m2, args.maximum_update_kg_m2)
        prediction = as_float_dict(state_values(state, static))
        history.append({
            "iteration": iteration,
            "bias_before_kg_m2": float(bias.detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "gradient": float(gradient.detach().cpu()),
            "gauss_newton_curvature": float(curvature.detach().cpu()),
            "raw_update_kg_m2": float(raw_update.detach().cpu()),
            "applied_update_kg_m2": float(update.detach().cpu()),
            "soil_water_layer_1": prediction["soil_water_layer_1"],
            "soil_water_layer_2": prediction["soil_water_layer_2"],
            "soil_temperature_layer_1": prediction["soil_temperature_layer_1"],
        })
        with torch.no_grad():
            bias.add_(update)
            bias.clamp_(min=lower_bias, max=upper_bias)
        del state, residual, loss, jacobian_tensor, gradient, curvature, raw_update, update
    final_bias = float(bias.detach().cpu())
    with torch.inference_mode():
        unperturbed_target_state = run_rollout(
            bias=torch.zeros((), dtype=DTYPE, device=device), nsteps=args.target_step,
            base_state=base_state, static=static, forcing_arrays=forcing_arrays,
            lat=lat, lon=lon, device=device, imported=imported,
        )
        biased_target_state = run_rollout(
            bias=torch.tensor(float(np.clip(args.initial_bias_kg_m2, lower_bias, upper_bias)), dtype=DTYPE, device=device),
            nsteps=args.target_step, base_state=base_state, static=static, forcing_arrays=forcing_arrays,
            lat=lat, lon=lon, device=device, imported=imported,
        )
        retrieved_target_state = run_rollout(
            bias=torch.tensor(final_bias, dtype=DTYPE, device=device), nsteps=args.target_step,
            base_state=base_state, static=static, forcing_arrays=forcing_arrays,
            lat=lat, lon=lon, device=device, imported=imported,
        )
        unperturbed_validation_state = run_rollout(
            bias=torch.zeros((), dtype=DTYPE, device=device), nsteps=args.validation_step,
            base_state=base_state, static=static, forcing_arrays=forcing_arrays,
            lat=lat, lon=lon, device=device, imported=imported,
        )
        biased_validation_state = run_rollout(
            bias=torch.tensor(float(np.clip(args.initial_bias_kg_m2, lower_bias, upper_bias)), dtype=DTYPE, device=device),
            nsteps=args.validation_step, base_state=base_state, static=static, forcing_arrays=forcing_arrays,
            lat=lat, lon=lon, device=device, imported=imported,
        )
        retrieved_validation_state = run_rollout(
            bias=torch.tensor(final_bias, dtype=DTYPE, device=device), nsteps=args.validation_step,
            base_state=base_state, static=static, forcing_arrays=forcing_arrays,
            lat=lat, lon=lon, device=device, imported=imported,
        )
    elapsed = time.perf_counter() - started
    target_evaluations = [
        evaluation_record("unperturbed restart", unperturbed_target_state, static, target_values),
        evaluation_record("prescribed biased restart", biased_target_state, static, target_values),
        evaluation_record("retrieved restart", retrieved_target_state, static, target_values),
    ]
    validation_evaluations = [
        evaluation_record("unperturbed restart", unperturbed_validation_state, static, validation_values),
        evaluation_record("prescribed biased restart", biased_validation_state, static, validation_values),
        evaluation_record("retrieved restart", retrieved_validation_state, static, validation_values),
    ]
    with (args.output / "fortran_guided_initial_state_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_result(
        history,
        validation_evaluations,
        args.output / "fortran_guided_initial_state_inversion.png",
        final_bias,
    )
    summary = {
        "experiment_type": "Fortran-guided initial-state inversion prototype; not observational calibration",
        "configuration": {
            "physics_timestep_s": 1800,
            "target_step": args.target_step,
            "target_day": args.target_step * 1800.0 / 86400.0,
            "validation_step": args.validation_step,
            "validation_day": args.validation_step * 1800.0 / 86400.0,
            "control": "initial top-layer soil-water mass bias",
            "control_unit": "kg m-2",
            "objective_fields": list(CONTROL_FIELDS),
            "objective_scales": CONTROL_SCALES.tolist(),
            "initial_bias_kg_m2": args.initial_bias_kg_m2,
            "retrieved_bias_kg_m2": final_bias,
            "bias_bounds_kg_m2": [lower_bias, upper_bias],
            "gauss_newton_iterations": args.iterations,
            "damping": args.damping,
            "maximum_update_kg_m2": args.maximum_update_kg_m2,
        },
        "column": {
            "index": column,
            "latitude_deg_n": float(lat[0]),
            "longitude_deg_e": float(lon[0]),
            "selection": "fixed warm, snow-free median-state rule based only on Fortran day-10 and day-20 fields" if args.column_index is None else "user-specified",
        },
        "target_day10": target_evaluations,
        "held_out_day20": validation_evaluations,
        "history": history,
        "elapsed_seconds": elapsed,
        "source": source_record(args.source),
        "inputs": {
            "case": str(args.case.resolve()),
            "data": str(args.data.resolve()),
            "fortran_reference": str(args.reference.resolve()),
        },
    }
    (args.output / "fortran_guided_initial_state_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({
        "column": summary["column"],
        "initial_bias_kg_m2": args.initial_bias_kg_m2,
        "retrieved_bias_kg_m2": final_bias,
        "elapsed_seconds": elapsed,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
