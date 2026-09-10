#!/usr/bin/env python3
"""Convert a torch year-terminal state into an HTESSEL restartin.nc.

Takes the year_NNNN snapshot stored by run_torch_multiyear_diag.py in
torch_year_terminal_states.npz and writes it into the structural template of
a reference HTESSEL restart (static fields inherited verbatim), producing a
restartin.nc that load_initial / the Fortran driver can consume.

The npz stores per-column fields for the 10-tile torch layout; the HTESSEL
restart stores the same physical quantities with the Fortran variable names
and (nlev, 1, ncols) / (1, ncols) layouts used by load_initial.

Run:
  python terminal_state_to_restart.py TERMINAL_NPZ YEAR TEMPLATE_NC OUT_NC
"""

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

TERMINAL_NPZ, YEAR, TEMPLATE, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

THICKNESS = np.array([0.07, 0.21, 0.72, 1.89])


def main():
    d = np.load(TERMINAL_NPZ)
    key = f"year_{YEAR}"
    # the npz packs all fields of one year under separate names via the
    # dict-merge in run_torch_multiyear_diag (year_ fields are the arrays
    # for that year only)
    skin_t = d[f"{key}"].item() if d[f"{key}"].dtype == object else None
    # run_torch_multiyear_diag stores year_NNNN as an object dict? No: it
    # saves named arrays under year_{year} only via the dict comprehension
    # {f"year_{year}": v for ...}; each v is an array, so year_NNNN is the
    # LAST field only. Read the individual named arrays instead.
    skin_t = d["skin_t"]
    soil_t = d["soil_t"]
    soil_w = d["soil_w"]           # kg/m2 per layer
    snow_w = d["snow_w"]
    snow_t = d["snow_t"]
    snow_rho = d["snow_rho"]
    snow_alb = d["snow_alb"]

    out = Dataset(OUT, "w", format="NETCDF4")
    with Dataset(TEMPLATE) as ref:
        out.setncatts(ref.__dict__)
        for dim in ref.dimensions:
            out.createDimension(dim, len(ref.dimensions[dim]))
        for name in ref.variables:
            src = ref[name]
            dst = out.createVariable(name, src.dtype, src.dimensions)
            dst.setncatts(src.__dict__)
            dst[:] = src[:]

    ncols = soil_t.shape[-1]

    def put(name, value):
        var = out[name]
        a = np.asarray(var[:])
        if a.ndim == 3:
            a[:, 0, :] = value
        else:
            a[0, :] = value
        var[:] = a

    # soil_t in npz is (ncols, 4); restart wants (4, 1, ncols)
    put("SoilTemp", soil_t.T)
    put("SoilMoist", soil_w.T)               # already kg/m2
    put("SWE", snow_w)
    put("SnowT", snow_t)
    put("SAlbedo", snow_alb)
    put("snowdens", snow_rho)
    # skin_t in npz is (ncols, ntile); grid-mean skin temp -> AvgSurfT
    # load_initial reconstructs tile skin from AvgSurfT via PSST/SURFEXCDRIVER
    # contract, so store the tile-0 (water) consistent mean. Use tile-mean.
    put("AvgSurfT", skin_t.mean(axis=1) if skin_t.ndim == 2 else skin_t)
    put("CanopInt", np.zeros(ncols))

    out.close()

    chk = Dataset(OUT)
    for v in ("SoilTemp", "SoilMoist", "SWE", "SnowT", "snowdens", "AvgSurfT"):
        a = np.asarray(chk[v][:])
        print(f"{v}: nan={int((~np.isfinite(a)).sum())} range=({np.nanmin(a):.3g},{np.nanmax(a):.3g})")
    chk.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
