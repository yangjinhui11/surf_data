#!/usr/bin/env python3
"""Annual water and energy-budget diagnostics from the model's own SurfDiag.

Runs the continuous 2018 integration (common ERA5-Land spun-up restart) with
``compute_diagnostics=True`` and accumulates the per-step water and energy
budget terms the model already computes into daily series plus terminal
maps, in the same layout as the manuscript's water_energy_budget npz.

Outputs (OUT_NPZ): day, mean_precip, mean_evaporation, mean_surface_runoff,
mean_subsurface_runoff, mean_storage_change, mean_water_residual,
rms_water_residual, max_abs_water_residual, mean_energy_residual,
mean_abs_energy_residual, max_abs_energy_residual, lat, lon,
cumulative_precip, cumulative_evaporation, cumulative_surface_runoff,
cumulative_subsurface_runoff, water_storage_change,
cumulative_water_residual, terminal_mean_energy_residual,
terminal_mean_abs_energy_residual.

Run on the server (GPU):
  python compute_budget_year.py --case CASE --data DATA --observations OBS \
      --source SRC --runner RUNNER --output OUT.npz --device cuda
"""

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.environ.get("SURF_RUNNER", ""))
sys.path.insert(0, os.environ.get("SURF_CODE", ""))
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
    state, static_full = load_initial(args.case / "restartin.nc", nvalid, args.device, lat, lon)
    config = SurfConfig(dtype=torch.float64, device=args.device,
                        differentiable=True, validate=False, use_farquhar=True,
                        use_ags=False, levgen=True, lelwtl=True, lelaiv=False,
                        vup_boundary_mode="offline", compute_diagnostics=True,
                        learnable_veg=PARAMETER_NAMES,
                        learnable_soil=("lambdadry", "lamsat1", "rcsoil"))
    model = LandSurface(config).to(args.device)
    fopts = SimpleNamespace(shift_flux=True, shift_state=False, shift_sw=False,
                            shift_lw=False, shift_precip=False, use_tendencies=True,
                            mu0_index="flux", lw_mode="net",
                            stress_u=None, stress_v=None, vup=None)
    area_w = torch.cos(torch.deg2rad(torch.as_tensor(lat, dtype=torch.float64,
                                                     device=args.device)))
    wsum = area_w.sum()

    def wmean(x):
        return float((x * area_w).sum() / wsum)

    series = {k: np.zeros(NDAYS) for k in
              ("mean_precip", "mean_evaporation", "mean_surface_runoff",
               "mean_subsurface_runoff", "mean_storage_change",
               "mean_water_residual", "rms_water_residual",
               "max_abs_water_residual", "mean_energy_residual",
               "mean_abs_energy_residual", "max_abs_energy_residual")}
    accum = {k: torch.zeros(nvalid, dtype=torch.float64, device=args.device) for k in
             ("precip", "evap", "sro", "ssro", "resid", "energy_resid", "energy_abs")}
    storage_start = None

    with torch.inference_mode():
        for day in range(NDAYS):
            day_acc = {k: torch.zeros(nvalid, dtype=torch.float64, device=args.device)
                       for k in ("precip", "evap", "sro", "ssro", "resid", "eres", "eabs")}
            for step_in_day in range(STEPS_PER_DAY):
                step = day * STEPS_PER_DAY + step_in_day
                fraction = model._tile_frac(static_full, config.n_tile, state=state)
                runtime = interpolated_runtime(forcing_full, step, DT_SECONDS,
                                               lat, lon, args.device)
                emission = model._surface_emissivity(state, static_full, fraction,
                                                     torch.float64, args.device)
                step_static = replace(static_full, emis=emission)
                forcing = make_forcing(runtime, 0, state, step_static, fopts,
                                       tile_frac=fraction)
                precipitation = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
                forcing = replace(forcing,
                                  rain=precipitation * (1.0 - runtime.precip_snow_phase),
                                  snow=precipitation * runtime.precip_snow_phase)
                state, diag = model(state, forcing, step_static)
                if storage_start is None:
                    storage_start = diag.water_storage.detach().clone()
                day_acc["precip"] += diag.precip_input.detach()
                day_acc["evap"] += diag.evaporation_input.detach()
                day_acc["sro"] += diag.surface_runoff.detach()
                day_acc["ssro"] += diag.subsurface_runoff.detach()
                day_acc["resid"] += diag.water_budget_residual.detach()
                day_acc["eres"] += diag.energy_budget_residual.detach()
                day_acc["eabs"] += diag.energy_budget_residual.detach().abs()

            series["mean_precip"][day] = wmean(day_acc["precip"])
            series["mean_evaporation"][day] = wmean(day_acc["evap"])
            series["mean_surface_runoff"][day] = wmean(day_acc["sro"])
            series["mean_subsurface_runoff"][day] = wmean(day_acc["ssro"])
            series["mean_storage_change"][day] = wmean(diag.water_storage_change.detach())
            series["mean_water_residual"][day] = wmean(diag.water_budget_residual.detach())
            r = diag.water_budget_residual.detach()
            series["rms_water_residual"][day] = float(
                torch.sqrt((r * r * area_w).sum() / wsum))
            series["max_abs_water_residual"][day] = float(r.abs().max())
            series["mean_energy_residual"][day] = wmean(diag.energy_budget_residual.detach())
            series["mean_abs_energy_residual"][day] = wmean(diag.energy_budget_residual.detach().abs())
            series["max_abs_energy_residual"][day] = float(diag.energy_budget_residual.detach().abs().max())

            for k_out, k_day in (("precip", "precip"), ("evap", "evap"),
                                 ("sro", "sro"), ("ssro", "ssro"),
                                 ("resid", "resid"), ("energy_resid", "eres"),
                                 ("energy_abs", "eabs")):
                accum[k_out] += day_acc[k_day]
            if (day + 1) % 30 == 0:
                print(json.dumps({"day": day + 1}), flush=True)

    out = {
        "step": np.arange(NDAYS) + 1,
        "day": np.arange(NDAYS) + 1,
        "lat": lat, "lon": lon,
        "cumulative_precip": accum["precip"].cpu().numpy(),
        "cumulative_evaporation": accum["evap"].cpu().numpy(),
        "cumulative_surface_runoff": accum["sro"].cpu().numpy(),
        "cumulative_subsurface_runoff": accum["ssro"].cpu().numpy(),
        "water_storage_change": (diag.water_storage.detach() - storage_start).cpu().numpy(),
        "cumulative_water_residual": accum["resid"].cpu().numpy(),
        "terminal_mean_energy_residual": accum["energy_resid"].cpu().numpy() / NDAYS,
        "terminal_mean_abs_energy_residual": accum["energy_abs"].cpu().numpy() / NDAYS,
    }
    out.update(series)
    np.savez(args.output, **out)
    print(json.dumps({"output": str(args.output),
                      "rms_water_residual_final": float(series["rms_water_residual"][-1]),
                      "terminal_water_resid_mean": float(accum["resid"].mean() / NDAYS)}))


if __name__ == "__main__":
    main()
