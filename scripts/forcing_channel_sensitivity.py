#!/usr/bin/env python3
"""Per-channel forcing sensitivity of daytime LST.

For each offline forcing channel, applies a single physically meaningful
perturbation (additive or multiplicative) and measures the resulting change
in domain-mean daytime land surface temperature over a strong-insolation
day. The resulting ranking quantifies which channels dominate daytime LST,
substantiating why the closure acts on the exchange air temperature and
absorbs the shortwave/radiative correction.

Channels (forcing vector y of Eq. (model-forcing)):
  tair      additive +1 K          (exchange air temperature)
  swdown    multiplicative x1.10   (downward shortwave)
  wind      multiplicative x1.10   (wind speed -> |V|)
  qair      multiplicative x1.10   (specific humidity)
  lwdown    additive +10 W/m2      (downward longwave)
  psurf     multiplicative x1.01   (surface pressure -> rho)
  precip    multiplicative x2.0    (precipitation, water only)

Protocol: one strong-insolation summer day from the spun-up restart, full
97,709-column domain, strict day-pass LST at the MODIS day-overpass step.
Baseline = unperturbed. Reports domain-mean day LST shift per channel.

Run on the server (GPU):
  python forcing_channel_sensitivity.py --case CASE --data DATA \
      --observations OBS --source SRC --runner RUNNER --output OUT.json \
      --day 181 --device cuda
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
FORCING_NAMES = ("tair_K", "qair_kg_kg", "psurf_Pa", "wind_m_s",
                 "swdown_W_m2", "lwdown_W_m2", "precip_kg_m2_s")
PARAMETER_NAMES = ("lai", "rsmin", "z0m")

# channel -> (kind, magnitude)
CHANNELS = {
    "baseline": (None, 1.0),
    "tair": ("add_t", 1.0),
    "swdown": ("mul_sw", 1.10),
    "wind": ("mul_wind", 1.10),
    "qair": ("mul_q", 1.10),
    "lwdown": ("add_lw", 10.0),
    "psurf": ("mul_p", 1.01),
    "precip": ("mul_precip", 2.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--day", type=int, default=181)
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
                        vup_boundary_mode="offline",
                        learnable_veg=PARAMETER_NAMES,
                        learnable_soil=("lambdadry", "lamsat1", "rcsoil"))
    model = LandSurface(config).to(args.device)
    fopts = SimpleNamespace(shift_flux=True, shift_state=False, shift_sw=False,
                            shift_lw=False, shift_precip=False, use_tendencies=True,
                            mu0_index="flux", lw_mode="net",
                            stress_u=None, stress_v=None, vup=None)
    area_w = torch.cos(torch.deg2rad(torch.as_tensor(lat, dtype=torch.float64,
                                                     device=args.device)))
    day_index = args.day - 1

    def run_day(channel):
        kind, mag = CHANNELS[channel]
        state = base_state
        day_skin = None
        for step_in_day in range(STEPS_PER_DAY):
            step = day_index * STEPS_PER_DAY + step_in_day
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
            if kind == "add_t":
                forcing = replace(forcing, t=forcing.t + mag)
            elif kind == "mul_sw":
                forcing = replace(forcing, fr_so=forcing.fr_so * mag)
            elif kind == "mul_wind":
                forcing = replace(forcing, u=forcing.u * mag, v=forcing.v * mag)
            elif kind == "mul_q":
                forcing = replace(forcing, q=forcing.q * mag)
            elif kind == "add_lw":
                forcing = replace(forcing, fr_th=forcing.fr_th + mag * DT_SECONDS)
            elif kind == "mul_p":
                forcing = replace(forcing, p_surf=forcing.p_surf * mag)
            elif kind == "mul_precip":
                forcing = replace(forcing, rain=forcing.rain * mag,
                                  snow=forcing.snow * mag)
            with torch.no_grad():
                state = model(state, forcing, step_static)[0]
            # day-pass skin at the nominal 10:30 local overpass (~step 21)
            if step_in_day == 21:
                day_skin = state.radiative_skin_t.detach().clone()
        return day_skin

    results = {}
    base = run_day("baseline")
    base_mean = float((base * area_w).sum() / area_w.sum())
    results["baseline_day_lst_K"] = base_mean
    for channel in CHANNELS:
        if channel == "baseline":
            continue
        skin = run_day(channel)
        shift = float(((skin - base) * area_w).sum() / area_w.sum())
        results[channel] = {"day_lst_shift_K": shift,
                            "perturbation": CHANNELS[channel]}
        print(json.dumps({"channel": channel, "shift_K": round(shift, 4)}), flush=True)

    results["day"] = args.day
    ranked = sorted(((k, v["day_lst_shift_K"]) for k, v in results.items()
                     if isinstance(v, dict)),
                    key=lambda kv: -abs(kv[1]))
    results["sensitivity_ranking"] = [{"channel": c, "shift_K": s} for c, s in ranked]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"ranking": [(c, round(s, 3)) for c, s in ranked]}))


if __name__ == "__main__":
    main()
