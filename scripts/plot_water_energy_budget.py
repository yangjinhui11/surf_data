"""Plot compact annual SURF water and energy-budget diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "manifests" / "water_energy_budget_1y_2018.npz"
OUT = ROOT / "manuscript" / "figures" / "annual_water_energy_budget.png"


def main() -> None:
    d = np.load(DATA)
    day = d["day"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(day, d["mean_precip"], label="Precipitation", lw=1.8)
    ax.plot(day, d["mean_evaporation"], label="Evaporation", lw=1.8)
    ax.plot(day, -d["mean_surface_runoff"], label="Surface runoff", lw=1.8)
    ax.plot(day, -d["mean_subsurface_runoff"], label="Subsurface runoff", lw=1.8)
    ax.plot(day, d["mean_storage_change"], label="Storage change", lw=1.8)
    ax.set_title("Cumulative water budget")
    ax.set_xlabel("Integration time (day)")
    ax.set_ylabel("Water mass (kg m$^{-2}$)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 1]
    ax.semilogy(day, d["rms_water_residual"], label="RMS")
    ax.semilogy(day, d["max_abs_water_residual"], label="Maximum")
    ax.set_title("Water-budget residual")
    ax.set_xlabel("Integration time (day)")
    ax.set_ylabel("Absolute residual (kg m$^{-2}$)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(day, d["mean_energy_residual"], label="Signed mean")
    ax.plot(day, d["mean_abs_energy_residual"], label="Mean absolute")
    ax.plot(day, d["max_abs_energy_residual"], label="Maximum absolute")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_title("Surface-energy diagnostic residual")
    ax.set_xlabel("Integration time (day)")
    ax.set_ylabel("Residual (W m$^{-2}$)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    sc = ax.scatter(
        d["lon"], d["lat"], c=d["cumulative_water_residual"],
        s=1.2, cmap="RdBu_r", vmin=-1e-11, vmax=1e-11,
        rasterized=True,
    )
    ax.set_title("Day-365 cumulative water residual")
    ax.set_xlabel("Longitude (degree)")
    ax.set_ylabel("Latitude (degree)")
    ax.set_xlim(float(d["lon"].min()), float(d["lon"].max()))
    ax.set_ylim(float(d["lat"].min()), float(d["lat"].max()))
    ax.grid(alpha=0.2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.88)
    cb.set_label("kg m$^{-2}$")

    fig.savefig(OUT, dpi=220)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
