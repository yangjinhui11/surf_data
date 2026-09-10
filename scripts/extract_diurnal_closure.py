#!/usr/bin/env python3
"""Domain-mean diurnal cycle of H, SW, and the closure anomaly dT.

Runs the same continuous 2018 integration protocol as the slab observables
archive, accumulating domain-mean surface sensible heat flux H (downward
positive), downward shortwave SW, and the closure's exchange-air anomaly
dt_air into local-time 3-hour bins. Output feeds the overlay figure that
compares the closure's own dT(local hour) curve against the attribution
bar profile.

This reuses the slab_observables_year machinery via a stripped driver.

Run on the server (GPU):
  python extract_diurnal_closure.py --case CASE --data DATA \
      --observations OBS --source SRC --runner RUNNER --output OUT_NPZ \
      --lambda-value 0.05 --tau-day-hours 6 --tau-night-hours 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Module-level imports below read SURF_CASE/SURF_CODE/SURF_DATA at import time.
_CASE = os.environ.get("ARG_CASE", "")
os.environ["SURF_CASE"] = _CASE or os.environ.get("SURF_CASE", "")
if "ARG_SRC" in os.environ:
    os.environ["SURF_CODE"] = os.environ["ARG_SRC"]
if "ARG_DATA" in os.environ:
    os.environ["SURF_DATA"] = os.environ["ARG_DATA"]
sys.path.insert(0, os.environ.get("SURF_RUNNER", ""))
from diagnostic_scripts_compare_realcol import make_forcing  # noqa: E402
from replay_full_domain_controlled import load_initial  # noqa: E402
from run_full_domain_offline import interpolated_runtime  # noqa: E402
from surf_pytorch import LandSurface, SurfConfig  # noqa: E402

DT_SECONDS = 1800.0
STEPS_PER_DAY = 48
NDAYS = 365
FORCING_NAMES = ("tair_K", "qair_kg_kg", "psurf_Pa", "wind_m_s",
                 "swdown_W_m2", "lwdown_W_m2", "precip_kg_m2_s")
PARAMETER_NAMES = ("lai", "rsmin", "z0m")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lambda-value", type=float, default=0.05)
    p.add_argument("--tau-day-hours", type=float, default=6.0)
    p.add_argument("--tau-night-hours", type=float, default=3.0)
    p.add_argument("--days", default="")  # e.g. "91..181" ; empty = full year
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["SURF_CASE"] = str(args.case)
    os.environ["SURF_CODE"] = str(args.source)
    os.environ["SURF_DATA"] = str(args.data)

    metadata = json.loads((args.case / "metadata.json").read_text())
    nvalid = int(metadata["nvalid"])
    lat = np.asarray(np.load(args.data / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(args.data / "lon.npy", mmap_mode="r"), dtype=np.float64)
    forcing_full = {n: np.load(args.data / f"{n}.npy", mmap_mode="r") for n in FORCING_NAMES}

    base_state, static_full = load_initial(args.case / "restartin.nc", nvalid,
                                           args.device, lat, lon)
    config = SurfConfig(dtype=torch.float64, device=args.device,
                        differentiable=True, validate=False, use_farquhar=True,
                        use_ags=False, levgen=True, lelwtl=True, lelaiv=False,
                        vup_boundary_mode="offline", learnable_veg=PARAMETER_NAMES,
                        learnable_soil=("lambdadry", "lamsat1", "rcsoil"))
    model = LandSurface(config).to(args.device)
    forcing_options = SimpleNamespace(
        shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
        shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
        stress_u=None, stress_v=None, vup=None)

    lam = args.lambda_value
    tau_day = torch.tensor(args.tau_day_hours * 3600.0, dtype=torch.float64, device=args.device)
    tau_night = torch.tensor(args.tau_night_hours * 3600.0, dtype=torch.float64, device=args.device)
    sw_ref = torch.tensor(50.0, dtype=torch.float64, device=args.device)
    dt_const = torch.as_tensor(DT_SECONDS, dtype=torch.float64, device=args.device)
    area_w = torch.cos(torch.deg2rad(torch.as_tensor(lat, dtype=torch.float64,
                                                     device=args.device)))

    # local-time bin of each column: 8 bins of 3 h by longitude
    lon_t = torch.as_tensor(lon, dtype=torch.float64, device=args.device)
    # bin index in 0..7 for the *current* UTC step is computed per step below

    day_range = range(NDAYS)
    sums_h = torch.zeros(8, dtype=torch.float64, device=args.device)
    sums_sw = torch.zeros(8, dtype=torch.float64, device=args.device)
    sums_dt = torch.zeros(8, dtype=torch.float64, device=args.device)
    counts = torch.zeros(8, dtype=torch.float64, device=args.device)

    state = base_state
    dt_air = torch.zeros(nvalid, dtype=torch.float64, device=args.device)
    from dataclasses import replace
    t0 = time.perf_counter()

    for day in day_range:
        for step_in_day in range(STEPS_PER_DAY):
            step = day * STEPS_PER_DAY + step_in_day
            fraction = model._tile_frac(static_full, config.n_tile, state=state)
            runtime = interpolated_runtime(forcing_full, step, DT_SECONDS,
                                           lat, lon, args.device)
            emission = model._surface_emissivity(state, static_full, fraction,
                                                 torch.float64, args.device)
            step_static = replace(static_full, emis=emission)
            forcing = make_forcing(runtime, 0, state, step_static,
                                   forcing_options, tile_frac=fraction)
            precipitation = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
            forcing = replace(forcing,
                              rain=precipitation * (1.0 - runtime.precip_snow_phase),
                              snow=precipitation * runtime.precip_snow_phase)
            forcing = replace(forcing, t=forcing.t + dt_air)
            with torch.no_grad():
                state = model(state, forcing, step_static)[0]
                heat = (fraction * model._last_pahfsti).sum(dim=1).detach().clone()
                sw = runtime.swdown_W_m2[1].clamp(min=0.0)
                weight = sw / (sw + sw_ref)
                beta = torch.exp(-dt_const * (weight / tau_day + (1.0 - weight) / tau_night))
                # guard against diverged columns BEFORE the recurrence: their
                # fluxes are non-physical and must not enter dt_air. Divergence
                # shows as astronomically large |H|, so cap by magnitude, not
                # by isfinite (a finite 1e30 flux still poisons the mean).
                heat = torch.where(torch.isfinite(heat) & (heat.abs() < 1.0e4),
                                   heat, torch.zeros_like(heat))
                dt_air = beta * dt_air + (1.0 - beta) * (-lam * heat)
                # dt_air beyond a few hundred K is likewise divergent
                dt_air = torch.where(torch.isfinite(dt_air) & (dt_air.abs() < 1.0e3),
                                     dt_air, torch.zeros_like(dt_air))

                # local-time bin (3-hour) for this UTC step, per column
                utc_hour = (step_in_day * DT_SECONDS) / 3600.0
                local_hour = (utc_hour + lon_t / 15.0) % 24.0
                bin_idx = (local_hour / 3.0).long().clamp(0, 7)
                finite = torch.isfinite(heat) & torch.isfinite(dt_air)
                for b in range(8):
                    m = (bin_idx == b) & finite
                    if not m.any():
                        continue
                    wb = area_w[m]
                    wsum = wb.sum()
                    sums_h[b] += (heat[m] * wb).sum()
                    sums_sw[b] += (sw[m] * wb).sum()
                    sums_dt[b] += (dt_air[m] * wb).sum()
                    counts[b] += wsum
        if (day + 1) % 30 == 0:
            print(json.dumps({"day": day + 1, "elapsed_s": round(time.perf_counter() - t0, 1)}),
                  flush=True)

    means = {
        "h_W_m2": (sums_h / counts).cpu().numpy(),
        "sw_W_m2": (sums_sw / counts).cpu().numpy(),
        "dt_air_K": (sums_dt / counts).cpu().numpy(),
        "counts": counts.cpu().numpy(),
        "lambda": lam, "tau_day_h": args.tau_day_hours, "tau_night_h": args.tau_night_hours,
        "days": [int(day_range[0]), int(day_range[-1])],
    }
    np.savez(args.output, **means)
    print(json.dumps({"output": str(args.output),
                      "dt_air_K": means["dt_air_K"].round(2).tolist()}))


if __name__ == "__main__":
    main()
