#!/usr/bin/env python3
"""Construct an HTESSEL restart initial state from ERA5-Land.

Maps the ERA5-Land 2004-01-01 land state onto the CMFD 97,709-column domain,
writing a restartin.nc with the same structure as the existing HTESSEL
restart. Static fields (soil/vegetation tables, roughness, orography, land
cover, lake/glacier masks) are inherited unchanged from the reference restart
--- only the prognostic land state is replaced.

Variable mapping (ERA5-Land layer bounds match HTESSEL exactly):
  SoilTemp[K]        <- stl1..4                    (K, direct)
  SoilMoist[kg m-2]  <- swvl1..4 * RDAW * rho_w    (vol. fraction -> kg/m2)
  SWE[kg m-2]        <- sd * rho_w                 (m w.e. -> kg/m2)
  SnowT[K]           <- tsn                        (K, direct)
  SAlbedo[-]         <- asn                        (direct)
  snowdens[kg m-3]   <- rsn                        (direct)
  AvgSurfT[K]        <- skt                        (skin temperature)
  CanopInt[kg m-2]   <- 0                          (no canopy store in ERA5-Land)

RDAW = (0.07, 0.21, 0.72, 1.89) m, rho_w = 1000 kg m-3.
"""

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ERA5 = Path("/data/yangjinhui/surf_pytorch/era5land/era5land_20040101_00.nc")
REF = Path("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
           "modis_validation_1y_2018_spunup_20260822/restartin_spunup.nc")
OUT = Path("/data/yangjinhui/surf_pytorch/era5land/restartin_era5land_2004.nc")
COLS = Path("/data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset")

RDAW = np.array([0.07, 0.21, 0.72, 1.89])
RHO_W = 1000.0


def main():
    lat = np.load(COLS / "lat.npy", mmap_mode="r")
    lon = np.load(COLS / "lon.npy", mmap_mode="r")
    ncols = lat.size

    era = Dataset(ERA5)
    elat = np.asarray(era["latitude"][:])
    elon = np.asarray(era["longitude"][:])
    # nearest-grid index of each CMFD column in the ERA5-Land 0.1-deg grid
    lat_idx = np.rint((elat[0] - lat) / 0.1).astype(int)
    lon_idx = np.rint((lon - elon[0]) / 0.1).astype(int)
    lat_idx = np.clip(lat_idx, 0, len(elat) - 1)
    lon_idx = np.clip(lon_idx, 0, len(elon) - 1)

    def field(name):
        return np.asarray(era[name][0])[lat_idx, lon_idx].astype(np.float64)

    soil_t = np.stack([field(f"stl{k}") for k in (1, 2, 3, 4)])
    soil_w = np.stack([field(f"swvl{k}") for k in (1, 2, 3, 4)]) * RDAW[:, None] * RHO_W
    swe = field("sd") * RHO_W
    snow_t = field("tsn")
    salb = field("asn")
    sdens = field("rsn")
    skt = field("skt")

    era.close()

    # ERA5-Land masks ocean/ice-shelf columns with fill values (NaN after
    # netCDF4 masking). The CMFD domain includes those columns (landsea mask
    # is inherited), so fill them with physical defaults that keep SURF
    # finite. Where both the soil and the skin temperature are masked
    # (open-ocean columns with no skin temperature either), fall back to the
    # reference restart's own value for that column so no NaN survives.
    ref_tmp = Dataset(REF)
    ref_soil_t = np.asarray(ref_tmp["SoilTemp"][:, 0, :])
    ref_skt = np.asarray(ref_tmp["AvgSurfT"][0, :])
    ref_tmp.close()

    for k in range(4):
        bad = ~np.isfinite(soil_t[k])
        soil_t[k][bad] = skt[bad]
        still = ~np.isfinite(soil_t[k])
        soil_t[k][still] = ref_soil_t[k][still]
        bad = ~np.isfinite(soil_w[k])
        soil_w[k][bad] = 0.0
    swe[~np.isfinite(swe)] = 0.0
    snow_t[~np.isfinite(snow_t)] = 273.15
    salb[~np.isfinite(salb)] = 0.5
    sdens[~np.isfinite(sdens)] = 100.0
    bad_skt = ~np.isfinite(skt)
    skt[bad_skt] = soil_t[0][bad_skt]
    still = ~np.isfinite(skt)
    skt[still] = ref_skt[still]

    ref = Dataset(REF)
    out = Dataset(OUT, "w", format="NETCDF4")
    out.setncatts(ref.__dict__)
    for dim in ref.dimensions:
        out.createDimension(dim, len(ref.dimensions[dim]))
    # copy every variable verbatim, then overwrite the prognostic state
    for name in ref.variables:
        src = ref[name]
        dst = out.createVariable(name, src.dtype, src.dimensions)
        dst.setncatts(src.__dict__)
        dst[:] = src[:]
    ref.close()

    def put(name, value):
        var = out[name]
        a = np.asarray(var[:])
        if a.ndim == 3:  # (nlev, 1, ncols)
            a[:, 0, :] = value
        else:            # (1, ncols)
            a[0, :] = value
        var[:] = a

    put("SoilTemp", soil_t)
    put("SoilMoist", soil_w)
    put("SWE", swe)
    put("SnowT", snow_t)
    put("SAlbedo", salb)
    put("snowdens", sdens)
    put("AvgSurfT", skt)
    put("CanopInt", np.zeros(ncols))

    # sensible bounds mirroring HTESSEL validity ranges (from the reference)
    out["SoilMoist"][:] = np.clip(out["SoilMoist"][:], 0.0, None)
    out["SWE"][:] = np.clip(out["SWE"][:], 0.0, None)
    out["snowdens"][:] = np.where(out["SWE"][:] > 0, out["snowdens"][:], 100.0)
    out["SAlbedo"][:] = np.clip(out["SAlbedo"][:], 0.0, 1.0)
    out["SnowT"][:] = np.where(out["SWE"][:] > 0, out["SnowT"][:], 273.15)

    out.close()

    chk = Dataset(OUT)
    for v in ("SoilTemp", "SoilMoist", "SWE", "SnowT", "snowdens", "AvgSurfT"):
        a = np.asarray(chk[v][:])
        print(f"{v}: range=({np.nanmin(a):.3g},{np.nanmax(a):.3g}) mean={np.nanmean(a):.3g}")
    chk.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
