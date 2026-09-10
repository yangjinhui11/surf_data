#!/usr/bin/env python3
"""Continuous full-year 2018 integration of the slab closure with a fixed
lambda_s, sampling the MODIS observables with the same rule as the offline
runner (run_full_domain_offline.py under SURF_MODIS_TARGETS): each LST pass
at the nearest completed 30-min step to its UTC view time, snow fraction at
10:30 local solar time.  Initial state and forcing options are those of the
spun-up validation experiment; the output netCDF format matches
surf_modis_observables.nc so the prior file and this file can be compared
under identical protocols.
"""

from __future__ import annotations

import argparse
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
    STEPS_PER_DAY,
)

RHO_CP = 1.2 * 1005.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, default=0.05)
    parser.add_argument("--slab-depth", type=float, default=600.0)
    parser.add_argument("--tau-day-hours", type=float, default=None,
                        help="asymmetric closure: day relaxation time (h); requires "
                             "--tau-night-hours; default = tied tau = lambda_s*rho*cp*slab-depth")
    parser.add_argument("--tau-night-hours", type=float, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lam_value = args.lambda_value
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

    lat = np.asarray(np.load(args.data / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(args.data / "lon.npy", mmap_mode="r"), dtype=np.float64)
    forcing_full = {name: np.load(args.data / f"{name}.npy", mmap_mode="r") for name in FORCING_NAMES}

    # MODIS target steps, replicating the runner's rule exactly.
    with Dataset(args.observations) as observations:
        day_time = np.asarray(observations["lst_day_view_time_utc"][:], dtype=np.float64)
        night_time = np.asarray(observations["lst_night_view_time_utc"][:], dtype=np.float64)
        obs_lat = np.asarray(observations["lat"][:], dtype=np.float64)
        obs_lon = np.asarray(observations["lon"][:], dtype=np.float64)
        modis_time = np.asarray(observations["time"][:])
        modis_time_units = observations["time"].units
        modis_calendar = getattr(observations["time"], "calendar", "standard")
    if obs_lat.size != nvalid:
        raise ValueError("observation columns do not match the domain")
    if not np.allclose(obs_lat, lat, atol=1.0e-5) or not np.allclose(obs_lon, lon, atol=1.0e-5):
        raise ValueError("observation and forcing columns are not aligned")
    n_obs_days = day_time.shape[0]
    nsteps = n_obs_days * STEPS_PER_DAY
    day_offset = np.arange(n_obs_days, dtype=np.int64)[:, None] * STEPS_PER_DAY

    def target_steps(hours_utc: np.ndarray) -> np.ndarray:
        result = np.ones(hours_utc.shape, dtype=np.int64)
        finite = np.isfinite(hours_utc)
        result[finite] = np.rint(hours_utc[finite] * 3600.0 / DT_SECONDS).astype(np.int64)
        result = np.clip(result, 1, STEPS_PER_DAY)
        result = day_offset + result
        result[~finite] = -1
        return result

    snow_time = np.mod(10.5 - lon[None, :] / 15.0, 24.0)
    snow_time = np.broadcast_to(snow_time, day_time.shape)
    targets = {
        "lst_day": target_steps(day_time),
        "lst_night": target_steps(night_time),
        "snow_fraction": target_steps(np.array(snow_time, copy=True)),
    }

    state, static = load_initial(
        args.case / "restartin.nc", nvalid, args.device, lat, lon,
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
    scalar_beta = None if asym else math.exp(-DT_SECONDS / (lam_value * RHO_CP * args.slab_depth))
    cache: dict = {"dt_air": None}

    def advance(current_state, physical_step: int):
        fraction = model._tile_frac(static, config.n_tile, state=current_state)
        runtime = interpolated_runtime(
            forcing_full, physical_step, DT_SECONDS, lat, lon, args.device,
        )
        emission = model._surface_emissivity(
            current_state, static, fraction, torch.float64, args.device,
        )
        step_static = replace(static, emis=emission)
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
            heat = (fraction * model._last_pahfsti).sum(dim=1).detach()
            if scalar_beta is not None:
                beta = scalar_beta
            else:
                sw = runtime.swdown_W_m2[1].clamp(min=0.0)
                weight = sw / (sw + sw_ref_t)
                beta = torch.exp(-dt_const * (weight / tau_day_t
                                              + (1.0 - weight) / tau_night_t))
            previous = cache["dt_air"] if cache["dt_air"] is not None else torch.zeros_like(heat)
            cache["dt_air"] = beta * previous + (1.0 - beta) * (-lam_value * heat)
        return new_state

    outputs = {name: np.full((n_obs_days, nvalid), np.nan, dtype=np.float32) for name in targets}
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(nsteps):
            state = advance(state, step)
            completed = step + 1
            obs_day = step // STEPS_PER_DAY
            for name, target in targets.items():
                mask = target[obs_day] == completed
                if not mask.any():
                    continue
                tensor_mask = torch.as_tensor(mask, device=args.device)
                field = state.snow_frac if name == "snow_fraction" else state.radiative_skin_t
                outputs[name][obs_day, mask] = field[tensor_mask].detach().cpu().numpy().astype(np.float32)
            if completed % (STEPS_PER_DAY * 5) == 0:
                print(json.dumps({"completed_step": completed, "nsteps": nsteps,
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(args.output, "w", format="NETCDF4") as output:
        output.createDimension("time", n_obs_days)
        output.createDimension("column", nvalid)
        output.source = "slab closure, continuous full-year integration, MODIS observation operator"
        output.restart = str(args.case / "restartin.nc")
        output.physics_timestep_seconds = DT_SECONDS
        output.lambda_s = lam_value
        output.slab_depth_m = args.slab_depth
        output.tau_day_h = args.tau_day_hours if asym else float("nan")
        output.tau_night_h = args.tau_night_hours if asym else float("nan")
        output.time_matching = "nearest completed physics step to each MODIS UTC view time"
        output.lst_operator = "SURF native grid-mean radiative_skin_t (AvgSurfT)"
        output.snow_operator = "prognostic SURF snow fraction at 10:30 local solar time"
        time_variable = output.createVariable("time", modis_time.dtype, ("time",))
        time_variable.units = modis_time_units
        time_variable.calendar = modis_calendar
        time_variable[:] = modis_time
        output.createVariable("lat", "f4", ("column",))[:] = lat.astype(np.float32)
        output.createVariable("lon", "f4", ("column",))[:] = lon.astype(np.float32)
        chunks = (1, min(16384, nvalid))
        for name, values in outputs.items():
            variable = output.createVariable(name, "f4", ("time", "column"), zlib=True,
                                             complevel=4, chunksizes=chunks, fill_value=np.nan)
            variable.units = "K" if name.startswith("lst_") else "1"
            variable[:] = values
    print(json.dumps({"done": str(args.output),
                      "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
