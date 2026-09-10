#!/usr/bin/env python3
"""Replot modis_parameter_identifiability.png from the combined JSON.

Same layout as combine_modis_identifiability.py, but the two seasonal panels
are labelled by season (spring-summer / autumn-winter) instead of the
hypothetical calibration/validation roles, which were the reverse of the
actual training protocol used elsewhere in the paper.

Run:  python replot_identifiability.py COMBINED_JSON OUT_PNG
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def correlation(matrix: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    denominator = np.outer(scale, scale)
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0.0)


def main() -> None:
    combined = json.loads(Path(sys.argv[1]).read_text())
    out_png = Path(sys.argv[2])

    # In the combined product, "calibration" = spring-summer sample days
    # (91/181) and "validation" = autumn-winter (271/361); the actual
    # calibration protocol trains on autumn-winter and holds out
    # spring-summer, so the panels are labelled by season.
    spring_summer = np.asarray(combined["information_calibration"], dtype=np.float64)
    autumn_winter = np.asarray(combined["information_validation"], dtype=np.float64)
    results = combined["results"]

    labels = ("LAI", r"$r_{s,min}$", r"$z_{0m}$")
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.6), constrained_layout=True)
    for axis, matrix, title in (
        (axes[0, 0], spring_summer, "Spring-summer information correlation"),
        (axes[0, 1], autumn_winter, "Autumn-winter information correlation"),
    ):
        values = correlation(matrix)
        image = axis.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        axis.set_xticks(range(3), labels=labels)
        axis.set_yticks(range(3), labels=labels)
        axis.set_title(title)
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes[0, :].tolist(), label="normalized information", shrink=0.8)

    ss_eigenvalues = np.asarray(combined["spectrum_calibration"]["eigenvalues"])
    aw_eigenvalues = np.asarray(combined["spectrum_validation"]["eigenvalues"])
    axes[1, 0].semilogy(range(1, 4), ss_eigenvalues, "o-", label="spring-summer")
    axes[1, 0].semilogy(range(1, 4), aw_eigenvalues, "s-", label="autumn-winter")
    axes[1, 0].set_xticks(range(1, 4))
    axes[1, 0].set_xlabel("information eigenmode")
    axes[1, 0].set_ylabel("eigenvalue")
    axes[1, 0].set_title("Identifiability spectrum")
    axes[1, 0].grid(color="0.88")
    axes[1, 0].legend()

    gradient_rows = []
    gradient_labels = []
    for result in results:
        for period in ("day", "night"):
            gradient_rows.append(result["periods"][period]["loss_gradient_log_scale"])
            gradient_labels.append(f"day {result['observation_day_index'] + 1} {period}")
    gradient_values = np.asarray(gradient_rows)
    gradient_scale = np.max(np.abs(gradient_values))
    gradient_image = axes[1, 1].imshow(
        gradient_values, cmap="PuOr_r", vmin=-gradient_scale, vmax=gradient_scale,
        aspect="auto",
    )
    axes[1, 1].set_xticks(range(3), labels=labels)
    axes[1, 1].set_yticks(range(len(gradient_labels)), labels=gradient_labels, fontsize=7)
    axes[1, 1].set_title("Gradient of area-weighted LST loss")
    figure.colorbar(gradient_image, ax=axes[1, 1], label=r"$\partial \mathcal{J}/\partial\log s$")
    figure.savefig(out_png, dpi=300, bbox_inches="tight")
    print("wrote", out_png)


if __name__ == "__main__":
    main()
