#!/usr/bin/env python3
"""Convert the final torch spin-up state into an HTESSEL restartin.nc.

Reads the LAST snapshot of torch_offline_daily.npz from a chained spin-up
(whose fields follow the torch 10-tile layout) and writes it into the
structural template of a reference HTESSEL restart, so the Fortran driver
and load_initial can consume it as the unified initial state.

Run:
  python spinup_final_to_restart.py SPINUP_CASE TEMPLATE_NC OUT_NC
"""

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

CASE, TEMPLATE, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]


def main():
    d = np.load(CASE / "torch_offline_daily.npz")
    i = -1  # last snapshot = final equilibrium state
    skin_t = d["skin_t"][i]      # (ncols, ntile)
    soil_t = d["soil_t"][i]      # (ncols, 4)
    soil_w = d["soil_w"][i]      # (ncols, 4) kg/m2
    snow_w = d["snow_w"][i]
    snow_t = d["snow_t"][i]
    snow_rho = d["snow_rho"][i]
    snow_alb = d["snow_alb"][i]
    ncols = soil_t.shape[0]

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

    def put(name, value):
        var = out[name]
        a = np.asarray(var[:])
        if a.ndim == 3:
            a[:, 0, :] = value
        else:
            a[0, :] = value
        var[:] = a

    # The SURF snow scheme keeps SWE non-negative by returning the negative
    # (pre-clipping) snow mass to the soil water as PEMSSN, then zeroing the
    # snow store (water-conserving). Mirror that here rather than a bare
    # clip: add any negative snow mass to the top-layer soil water, then
    # zero SWE. Only a handful of columns are affected.
    neg = snow_w < 0.0
    soil_w = soil_w.copy()
    if neg.any():
        soil_w[neg, 0] = soil_w[neg, 0] + snow_w[neg]  # conserve water
        snow_w = np.where(neg, 0.0, snow_w)

    put("SoilTemp", soil_t.T)
    put("SoilMoist", np.clip(soil_w, 0.0, None).T)
    put("SWE", snow_w)
    put("SnowT", snow_t)
    put("SAlbedo", snow_alb)
    put("snowdens", snow_rho)
    put("AvgSurfT", skin_t.mean(axis=1))
    put("CanopInt", np.zeros(ncols))
    out.close()

    chk = Dataset(OUT)
    for v in ("SoilTemp", "SoilMoist", "SWE", "SnowT", "snowdens", "AvgSurfT"):
        a = np.asarray(chk[v][:])
        print(f"{v}: nan={int((~np.isfinite(a)).sum())} range=({np.nanmin(a):.3g},{np.nanmax(a):.3g}) mean={np.nanmean(a):.3g}")
    chk.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
