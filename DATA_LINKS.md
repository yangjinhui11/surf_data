# External datasets used by this study (given by reference)

The large raw driving datasets are public and are not duplicated in this
archive. The preprocessing scripts that map them onto the 97,709-column domain
are part of the accompanying software (available from the corresponding author upon request).

## CMFD — China Meteorological Forcing Dataset (raw forcing, all experiments)

- He, J., Yang, K., Tang, W., Lu, H., Qin, J., Chen, Y., and Li, X.: The first
  high-resolution meteorological forcing dataset for land process studies over
  China, Scientific Data, 7, 25, https://doi.org/10.1038/s41597-020-0369-y, 2020.
- Dataset (0.1°, 3-hourly, 1979–2018; this study uses 2004–2018):
  figshare collection https://doi.org/10.6084/m9.figshare.c.4557599
  (also distributed by the National Tibetan Plateau / Third Pole Environment
  Data Center, https://data.tpdc.ac.cn).

## ERA5-Land (initial-state construction only)

- Copernicus Climate Data Store, dataset "reanalysis-era5-land",
  https://doi.org/10.24381/cds.e2161bac.
- Only the 1 January 2005 state is used; the mapping onto the CMFD domain is
  performed by the accompanying software (available from the corresponding
  author upon request), and the mapped restart is included
  in this archive (`initial_state/restartin_era5land_2005.nc`), as is the
  resulting spun-up state (`initial_state/restartin_era5land_2009_eq.nc`).

## MODIS (independent evaluation and calibration targets)

- MOD11C1 Collection 6.1 (Terra daily daytime/nighttime land surface
  temperature, 0.05° CMG): NASA LAADS DAAC,
  https://doi.org/10.5067/MODIS/MOD11C1.061.
- MOD10C1 Collection 6.1 (Terra daily fractional snow cover, 0.05° CMG):
  NASA LAADS DAAC, https://doi.org/10.5067/MODIS/MOD10C1.061.
- The 2018 subsets sampled at the CMFD column centres are included in this
  archive (`observations/modis_cmg/`); re-sampling other years uses the
  preprocessing scripts of the accompanying software.

## Reference implementation

- OpenIFS SURF (IFS cycle 48r1) offline configuration: native source subject to
  ECMWF distribution constraints (https://www.ecmwf.int/en/research/projects/openifs).
  The offline Fortran reference outputs used in every comparison of the paper
  are included in `fortran_reference/`.
