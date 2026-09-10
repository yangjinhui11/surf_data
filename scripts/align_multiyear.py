# -*- coding: utf-8 -*-
"""Robust torch<->Fortran alignment for the chained multi-year runs.

Hard-won rules (do NOT re-derive loosely):
  * The chained torch run integrates each calendar year for its TRUE number of
    steps: 17520 for a 365-day year, 17568 for a leap year (forcing records
    ntime*10800/dt, with ntime=2920 or 2928).  global_step accumulates these.
  * The Fortran reference integrates the SAME per-year step count (NSTOP =
    17520 or 17568).  Its per-year ``o_gg.nc`` only archives every 480 steps up
    to the last multiple of 480, so the archived last record is NOT the year
    terminal.  The true 365/366-day terminal state is ``restartout.nc``.
  * Therefore the year-end alignment is
        torch global step == cumulative sum of steps_in_year(year)
        <-> Fortran  <year>/restartout.nc
    and the year-1 (2009) terminal must match near-bitwise (shared spin-up
    initial state), which serves as a built-in alignment self-check.

Leap years in 2009-2018: 2012, 2016.
"""
from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

FORCING_DT = 10800.0
PHYSICS_DT = 1800.0


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def steps_in_year(year: int) -> int:
    """Physics steps in one calendar year at 1800 s (17520 or 17568)."""
    return 17568 if is_leap(year) else 17520


def year_end_global_steps(years) -> dict[int, int]:
    """Cumulative torch global step at the terminal of each year."""
    cum = 0
    out = {}
    for y in years:
        cum += steps_in_year(y)
        out[y] = cum
    return out


def torch_terminal_index(steps: np.ndarray, year: int, years) -> int:
    """Row index in the torch daily npz for the terminal of ``year``.

    Raises if the expected global step is not archived (fail loud, never
    silently misalign)."""
    gstep = year_end_global_steps(years)[year]
    steps = np.asarray(steps, dtype=np.int64)
    hits = np.flatnonzero(steps == gstep)
    if hits.size == 0:
        raise ValueError(
            f"year {year}: expected torch terminal global step {gstep} not in "
            f"archived steps (range {steps.min()}..{steps.max()}).  Refusing to "
            f"approximate -- check the run archived every year terminal."
        )
    return int(hits[0])


def fortran_terminal(year: int, froot: str, name: str) -> np.ndarray:
    """Fortran true 365/366-day terminal field from restartout.nc."""
    with Dataset(f"{froot}/{year}/restartout.nc") as d:
        return np.asarray(d.variables[name][:], dtype=np.float64)


def selfcheck_year1(torch_swe: np.ndarray, torch_steps: np.ndarray,
                    froot: str, year0: int = 2009, tol=1e-3) -> bool:
    """Year-1 shares the spin-up initial state, so its terminal must match the
    Fortran restartout almost exactly.  If not, the alignment is wrong."""
    years = list(range(year0, year0 + 10))
    i = torch_terminal_index(np.asarray(torch_steps), year0, years)
    t = float(np.asarray(torch_swe[i]).mean())
    f = float(fortran_terminal(year0, froot, "SWE").mean())
    ok = abs(t - f) <= tol * max(1.0, abs(f))
    print(f"[alignment self-check] {year0} terminal domain-mean SWE: "
          f"torch={t:.6f} fortran={f:.6f} -> {'OK' if ok else 'MISALIGNED'}")
    if not ok:
        raise ValueError("alignment self-check failed: do not trust downstream metrics")
    return True
