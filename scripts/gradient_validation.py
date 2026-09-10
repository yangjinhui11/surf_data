#!/usr/bin/env python3
"""Verify multi-step SURF autograd derivatives against finite differences.

The experiment intentionally uses a deterministic 24 h diurnal forcing and a
small land-only batch. It validates the differentiated SURF trajectory itself,
without conflating derivative accuracy with CMFD forcing interpolation or
offline file I/O. Results are written as compact CSV/JSON/PNG artifacts.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from surf_pytorch import IdealizedDiurnalConfig, LandSurface, SurfConfig, build_idealized_diurnal


OUTDIR = Path(os.environ.get("SURF_GRADIENT_OUTPUT", "."))
DTYPE = torch.float64
EPSILONS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)
VEG_CLASS = 3


def build_case():
    cfg = IdealizedDiurnalConfig(
        n_grid=4,
        n_steps=24,
        dt_seconds=3600.0,
        lat_deg=45.0,
        lon_deg=0.0,
        day_of_year=172,
        tair_mean=290.0,
        tair_amp=8.0,
        sw_peak=800.0,
        soil_t_init=285.0,
        soil_w_init=0.30,
        skin_t_init=288.0,
        dtype=DTYPE,
    )
    return build_idealized_diurnal(cfg)


def run_loss(kind: str, value: torch.Tensor | float, *, differentiable: bool) -> torch.Tensor:
    """Run a 24-step trajectory and return one physically interpretable loss."""
    forcing, static, state = build_case()
    model = LandSurface(SurfConfig(
        dtype=DTYPE,
        differentiable=differentiable,
        validate=False,
        learnable_veg=("rvlai",) if kind == "rvlai" else (),
    ))

    if kind == "tair_offset":
        forcing = [replace(f, t=f.t + value) for f in forcing]
    elif kind == "soil_w0_offset":
        top_soil = state.soil_w_liq.clone()
        top_soil[:, 0] = top_soil[:, 0] + value
        state = replace(state, soil_w_liq=top_soil)
    elif kind == "rvlai":
        rvlai = dict(model.veg_table.named_parameters())["rvlai"]
        with torch.no_grad():
            rvlai[VEG_CLASS] += float(value)
    else:
        raise ValueError(kind)

    evap_integral = torch.zeros((), dtype=DTYPE)
    for f in forcing:
        state, diag = model(state, f, static)
        evap_integral = evap_integral + diag.e_flux.mean() * f.dt.mean()

    if kind == "tair_offset":
        return state.skin_t.mean()
    if kind == "soil_w0_offset":
        return state.soil_w_liq[:, 0].mean()
    return evap_integral


def autograd_derivative(kind: str) -> tuple[float, float]:
    if kind == "rvlai":
        forcing, static, state = build_case()
        model = LandSurface(SurfConfig(
            dtype=DTYPE, differentiable=True, validate=False, learnable_veg=("rvlai",),
        ))
        rvlai = dict(model.veg_table.named_parameters())["rvlai"]
        evap_integral = torch.zeros((), dtype=DTYPE)
        for f in forcing:
            state, diag = model(state, f, static)
            evap_integral = evap_integral + diag.e_flux.mean() * f.dt.mean()
        gradient = torch.autograd.grad(evap_integral, rvlai)[0][VEG_CLASS]
        return float(evap_integral.detach()), float(gradient.detach())

    control = torch.zeros((), dtype=DTYPE, requires_grad=True)
    loss = run_loss(kind, control, differentiable=True)
    gradient = torch.autograd.grad(loss, control)[0]
    return float(loss.detach()), float(gradient.detach())


def central_difference(kind: str, epsilon: float) -> float:
    # Finite differences must perturb the exact same differentiable forward
    # map used by autograd; inference mode may select numerically equivalent
    # but graph-free helper paths.
    plus = float(run_loss(kind, epsilon, differentiable=True))
    minus = float(run_loss(kind, -epsilon, differentiable=True))
    return (plus - minus) / (2.0 * epsilon)


def main() -> None:
    if os.environ.get("SURF_GRADIENT_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    labels = {
        "tair_offset": ("air-temperature offset", "final mean skin temperature", "K K$^{-1}$"),
        "soil_w0_offset": ("initial top-soil water offset", "final top-soil water", "m$^3$ m$^{-3}$ per m$^3$ m$^{-3}$"),
        "rvlai": ("vegetation LAI-table value", "24 h cumulative evaporation", "kg m$^{-2}$ per LAI"),
    }
    rows: list[dict[str, float | str]] = []
    summary: dict[str, dict[str, float | str]] = {}
    for kind in labels:
        loss, ad = autograd_derivative(kind)
        for epsilon in EPSILONS:
            fd = central_difference(kind, epsilon)
            relative_error = abs(ad - fd) / max(abs(fd), 1.0e-12)
            rows.append({
                "control": kind,
                "epsilon": epsilon,
                "loss_at_control": loss,
                "autograd": ad,
                "central_difference": fd,
                "relative_error": relative_error,
            })
        best = min((row for row in rows if row["control"] == kind), key=lambda row: row["relative_error"])
        summary[kind] = {
            "control_label": labels[kind][0],
            "loss_label": labels[kind][1],
            "unit": labels[kind][2],
            "loss_at_control": loss,
            "autograd": ad,
            "best_epsilon": best["epsilon"],
            "best_central_difference": best["central_difference"],
            "best_relative_error": best["relative_error"],
        }

    with (OUTDIR / "gradient_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUTDIR / "gradient_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(11.3, 3.5), constrained_layout=True)
    for ax, kind in zip(axes, labels):
        selected = [row for row in rows if row["control"] == kind]
        eps = np.array([float(row["epsilon"]) for row in selected])
        errors = np.array([float(row["relative_error"]) for row in selected])
        ax.loglog(eps, errors, "o-", color="#1976a2", lw=2)
        ax.set_title(labels[kind][0])
        ax.set_xlabel("Finite-difference step")
        ax.grid(True, which="both", alpha=0.28)
        ax.text(0.04, 0.06, f"best = {summary[kind]['best_relative_error']:.2e}", transform=ax.transAxes)
    axes[0].set_ylabel("Relative gradient error")
    figure.savefig(OUTDIR / "gradient_finite_difference_check.png", dpi=300, bbox_inches="tight")

    for kind, values in summary.items():
        print(
            f"{kind}: autograd={values['autograd']:.8e}, "
            f"best_fd={values['best_central_difference']:.8e}, "
            f"relative_error={values['best_relative_error']:.3e}"
        )


if __name__ == "__main__":
    main()
