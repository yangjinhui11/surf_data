# SURF differentiable PyTorch implementation — paper data and analysis scripts

Data package and analysis scripts accompanying the manuscript "A differentiable
PyTorch implementation of the OpenIFS land-surface physics" (submitted to
Geoscientific Model Development). The model implementation source code, tests,
and production run scripts live in a separate repository that is private until
publication; they are available from the corresponding author upon request.

## Layout

- `manuscript/` — manuscript LaTeX sources, figures, and the compiled PDF.
- `experiments/` — archived result records (metrics, histories, logs, figures)
  of every experiment: controlled and autonomous Fortran–PyTorch equivalence,
  decadal chained integration, gradient verification, sensitivity maps,
  initial-state retrieval, MODIS identifiability, twin and real calibration,
  pseudo-forcing attribution, surface-layer closure training, and the
  performance benchmarks.
- `experiments/results_controlled/` — comparison records of the 30-day
  controlled boundary-driven experiment.
- `fortran_reference/` — offline Fortran reference outputs: 30-day controlled
  experiment, one-year 2018 integration, and per-year terminal states
  (`restartout.nc`) of the 2009–2018 chained integration. The native OpenIFS
  source code is subject to ECMWF distribution constraints and is not included.
- `observations/` — MODIS 2018 subset descriptor (`modis_cmg/*.json`).
- `initial_state/` — spin-up run log and summaries.
- `scripts/` — analysis, comparison, and figure-generation scripts.
- `manifests/` — per-experiment manifests with source-file SHA-256 hashes.
- `REPRODUCIBILITY.md` — the authoritative reproduction chain.
- `DATA_LINKS.md` — the public raw driving datasets (CMFD, ERA5-Land, MODIS),
  given by reference.
- `MANIFEST.sha256` — SHA-256 digest of every file in the full archive
  (including the large files listed below).

## Large files (not in this repository)

The following files exceed GitHub's 100 MB per-file limit and are held in the
full project archive (Zenodo DOI to be assigned upon upload) and on the project
server; all are regenerable with the accompanying software:

- `initial_state/restartin_era5land_2005.nc` (134 MB) — ERA5-Land 2005 state
  mapped onto the 97,709-column CMFD domain (spin-up input).
- `initial_state/restartin_era5land_2009_eq.nc` (134 MB) — the common spun-up
  initial state of every experiment.
- `observations/modis_cmg/modis_cmg_2018_cmfd_columns.nc` (219 MB) — MODIS 2018
  LST and snow cover sampled at the CMFD column centres.
- `evaluation/surf_modis_observables.nc` (231 MB) and
  `surf_modis_observables_slab.nc` (221 MB) — year-long observation-operator
  products of the uncalibrated model (Table 5) and of the calibrated closure
  configuration (Table 6).
- `fortran_reference/*/o_gg.nc` (221–263 MB) and
  `fortran_reference/*/restartout.nc` (109 MB each) — Fortran reference
  trajectories and terminal states.
