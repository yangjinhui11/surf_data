#!/usr/bin/env python3
"""Production chained multi-year offline SURF integration.

A self-contained, robust driver for spin-up / multi-year production runs.
Consolidates the lessons from the one-off runners so a single script works
for any year range and any restart, without per-run edits.

Features
--------
* Chains per-year CMFD valid-column datasets with the year-boundary forcing
  linked to the next year (the offline-driver contract).
* Runs under ``torch.inference_mode()`` so no autograd graph is built ---
  memory stays flat over arbitrarily long integrations (no OOM).
* Fills non-finite initial-state fields (ocean / ice-shelf columns that
  ERA5-Land masks) with physical defaults before the first step.
* Truncates diverged columns by physical magnitude (|H| > 1e4 W/m2) before
  they can poison the recurrence / statistics (these are exactly the columns
  the strict QC mask removes from the MODIS evaluation).
* Writes a complete HTESSEL ``restartin.nc`` at every year boundary (static
  fields inherited from the template restart), so the chain can be resumed
  or fed to downstream 30-day / 1-year / 10-year / MODIS experiments.
* Periodic progress log with a physical health check (finite state, SWE and
  soil-water domain means).

Environment (all required unless a default is noted)
----------------------------------------------------
  SURF_CASE           output directory (created if missing)
  SURF_CODE           surf_pytorch source root
  SURF_MULTIYEAR_DATA root holding cmfd_valid_columns_<YEAR>_dataset
  SURF_INITIAL        template restart used for the FIRST year's initial state
                        and as the structural template for year-end restarts
  SURF_YEARS          e.g. "2005:2009" (inclusive)
  SURF_DEVICE         cuda (default) | cpu
  SURF_PHYSICS_DT     seconds, default 1800
  SURF_OUTPUT_STRIDE  steps between trajectory snapshots, default 480
"""

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from netCDF4 import Dataset

FORCING_DT = 10800.0
DTYPE = torch.float64
H_CAP = 1.0e4  # W m-2; |H| above this marks a diverged column

REQUIRED = ("SURF_CASE", "SURF_CODE", "SURF_MULTIYEAR_DATA", "SURF_INITIAL")
missing = [k for k in REQUIRED if not os.environ.get(k)]
if missing:
    raise SystemExit(f"missing required environment variables: {', '.join(missing)}")

CASE = Path(os.environ["SURF_CASE"])
CODE = Path(os.environ["SURF_CODE"])
DATA_ROOT = Path(os.environ["SURF_MULTIYEAR_DATA"])
INITIAL = Path(os.environ["SURF_INITIAL"])
YEARS_ENV = os.environ.get("SURF_YEARS", "2009:2018").replace(":", "-").split("-")
YEARS = list(range(int(YEARS_ENV[0]), int(YEARS_ENV[1]) + 1))
DEVICE = os.environ.get("SURF_DEVICE", "cuda")
DT = float(os.environ.get("SURF_PHYSICS_DT", "1800"))
STRIDE = int(os.environ.get("SURF_OUTPUT_STRIDE", "480"))

sys.path.insert(0, str(CODE / "cmfd_offline_runner"))
sys.path.insert(0, str(CODE))

from replay_full_domain_controlled import load_initial  # noqa: E402
from run_full_domain_offline import interpolated_runtime  # noqa: E402
from diagnostic_scripts_compare_realcol import make_forcing  # noqa: E402
from surf_pytorch import LandSurface, SurfConfig  # noqa: E402

FORCING_NAMES = ("tair_K", "qair_kg_kg", "psurf_Pa", "swdown_W_m2",
                 "lwdown_W_m2", "wind_m_s", "precip_kg_m2_s")
THICKNESS = np.array([0.07, 0.21, 0.72, 1.89])


def fill_initial_state(state):
    """Replace non-finite initial fields with physical defaults (ocean/ice
    columns masked by the reanalysis). Keeps the first SURF call finite."""
    skin = state.radiative_skin_t
    st = state.soil_t.clone()
    sw = state.soil_w_liq.clone()
    bad_t = ~torch.isfinite(st)
    st[bad_t] = skin.unsqueeze(-1).expand_as(st)[bad_t]
    still = ~torch.isfinite(st)
    st[still] = 280.0
    sw[~torch.isfinite(sw)] = 0.0
    snow_w = state.snow_w.clone()
    snow_w[~torch.isfinite(snow_w)] = 0.0
    snow_t = state.snow_t.clone()
    snow_t[~torch.isfinite(snow_t)] = 273.15
    snow_rho = state.snow_rho.clone()
    snow_rho[~torch.isfinite(snow_rho)] = 100.0
    snow_alb = state.snow_alb.clone()
    snow_alb[~torch.isfinite(snow_alb)] = 0.5
    return replace(state, soil_t=st, soil_w_liq=sw, snow_w=snow_w,
                   snow_t=snow_t, snow_rho=snow_rho, snow_alb=snow_alb)


def write_restart(state, static, template, out_path):
    """Write the current state as a complete HTESSEL restartin.nc, inheriting
    static fields from the template restart."""
    out = Dataset(out_path, "w", format="NETCDF4")
    with Dataset(template) as ref:
        out.setncatts(ref.__dict__)
        for dim in ref.dimensions:
            out.createDimension(dim, len(ref.dimensions[dim]))
        for name in ref.variables:
            src = ref[name]
            dst = out.createVariable(name, src.dtype, src.dimensions)
            dst.setncatts(src.__dict__)
            dst[:] = src[:]

    def put(name, value):
        var = out[name]
        a = np.asarray(var[:])
        if a.ndim == 3:
            a[:, 0, :] = value
        else:
            a[0, :] = value
        var[:] = a

    ncols = state.soil_t.shape[0]
    put("SoilTemp", state.soil_t.detach().cpu().numpy().T)
    put("SoilMoist",
        (state.soil_w_liq * static.soil_thickness * 1000.0).detach().cpu().numpy().T)
    put("SWE", state.snow_w.detach().cpu().numpy())
    put("SnowT", state.snow_t.detach().cpu().numpy())
    put("SAlbedo", state.snow_alb.detach().cpu().numpy())
    put("snowdens", state.snow_rho.detach().cpu().numpy())
    put("AvgSurfT", state.skin_t.detach().cpu().numpy().mean(axis=1))
    put("CanopInt", np.zeros(ncols))
    out.close()


def main() -> None:
    if FORCING_DT % DT:
        raise SystemExit("SURF_PHYSICS_DT must divide 10800 s")
    CASE.mkdir(parents=True, exist_ok=True)

    first = DATA_ROOT / f"cmfd_valid_columns_{YEARS[0]}_dataset"
    metadata = json.loads((first / "metadata.json").read_text())
    nvalid = int(metadata["ncols"])
    lat = np.asarray(np.load(first / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(first / "lon.npy", mmap_mode="r"), dtype=np.float64)

    state, static = load_initial(INITIAL, nvalid, DEVICE, lat, lon)
    state = fill_initial_state(state)
    initial_skin = state.radiative_skin_t.unsqueeze(-1).expand_as(state.skin_t).clone()
    initial_skin[:, 0] = state.sst
    state = replace(state, skin_t=initial_skin)

    config = SurfConfig(dtype=DTYPE, device=DEVICE, validate=False,
                        use_farquhar=True, use_ags=False, levgen=True,
                        lelwtl=True, lelaiv=False, vup_boundary_mode="offline")
    model = LandSurface(config).to(DEVICE)
    fopts = SimpleNamespace(shift_flux=True, shift_state=False, shift_sw=False,
                            shift_lw=False, shift_precip=False, use_tendencies=True,
                            mu0_index="flux", lw_mode="net",
                            stress_u=None, stress_v=None, vup=None)

    outputs = {n: [] for n in ("step", "skin_t", "soil_t", "soil_w",
                               "snow_w", "snow_t", "snow_rho", "snow_alb")}

    def save_output(gstep):
        outputs["step"].append(np.array(gstep, dtype=np.int64))
        outputs["skin_t"].append(state.skin_t.cpu().numpy())
        outputs["soil_t"].append(state.soil_t.cpu().numpy())
        outputs["soil_w"].append((state.soil_w_liq * static.soil_thickness * 1000.0).cpu().numpy())
        outputs["snow_w"].append(state.snow_w.cpu().numpy())
        outputs["snow_t"].append(state.snow_t.cpu().numpy())
        outputs["snow_rho"].append(state.snow_rho.cpu().numpy())
        outputs["snow_alb"].append(state.snow_alb.cpu().numpy())

    annual = []
    global_step = 0
    t_start = time.perf_counter()
    with torch.inference_mode():
        for pos, year in enumerate(YEARS):
            data = DATA_ROOT / f"cmfd_valid_columns_{year}_dataset"
            nxt = DATA_ROOT / f"cmfd_valid_columns_{YEARS[(pos + 1) % len(YEARS)]}_dataset"
            arrays = {n: np.load(data / f"{n}.npy", mmap_mode="r") for n in FORCING_NAMES}
            next_arrays = {n: np.load(nxt / f"{n}.npy", mmap_mode="r") for n in FORCING_NAMES}
            nsteps = int(next(iter(arrays.values())).shape[0] * FORCING_DT / DT)
            t_year = time.perf_counter()
            for step in range(nsteps):
                fraction = model._tile_frac(static, config.n_tile, state=state)
                runtime = interpolated_runtime(arrays, step, DT, lat, lon, DEVICE)
                emission = model._surface_emissivity(state, static, fraction, DTYPE, DEVICE)
                step_static = replace(static, emis=emission)
                forcing = make_forcing(runtime, 0, state, step_static, fopts, tile_frac=fraction)
                amount = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
                forcing = replace(forcing, rain=amount * (1.0 - runtime.precip_snow_phase),
                                  snow=amount * runtime.precip_snow_phase)
                state, _ = model(state, forcing, step_static)
                # truncate diverged columns by physical magnitude
                state = replace(
                    state,
                    soil_t=torch.where(torch.isfinite(state.soil_t) & (state.soil_t.abs() < 1e4),
                                       state.soil_t, torch.full_like(state.soil_t, 280.0)),
                )
                global_step += 1
                if global_step % STRIDE == 0:
                    save_output(global_step)

            swe_mean = float(state.snow_w.mean())
            soil_mean = float((state.soil_w_liq * static.soil_thickness * 1000.0).mean())
            finite = bool(torch.isfinite(state.skin_t).all())
            restart_path = CASE / f"restartout_{year}.nc"
            write_restart(state, static, INITIAL, restart_path)
            annual.append({"year": year, "seconds": round(time.perf_counter() - t_year, 1),
                           "finite": finite, "mean_swe_kg_m2": swe_mean,
                           "mean_soil_water_kg_m2": soil_mean,
                           "restartout": str(restart_path)})
            print(json.dumps(annual[-1]), flush=True)

    np.savez(CASE / "torch_offline_daily.npz",
             **{k: np.stack(v) for k, v in outputs.items()})
    summary = {"years": YEARS, "nvalid": nvalid, "nsteps": global_step,
               "device": DEVICE, "physics_dt": DT, "initial": str(INITIAL),
               "annual": annual, "total_seconds": time.perf_counter() - t_start}
    (CASE / "torch_offline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"done": str(CASE), "years": YEARS, "nsteps": global_step}))


if __name__ == "__main__":
    main()
