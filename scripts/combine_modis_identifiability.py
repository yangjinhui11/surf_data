#!/usr/bin/env python3
"""Combine column-block MODIS identifiability results and plot diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def spectrum(matrix: np.ndarray) -> dict[str, object]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    threshold = eigenvalues[0] * 1.0e-12 if eigenvalues[0] > 0.0 else 0.0
    positive = eigenvalues[eigenvalues > threshold]
    condition = None if positive.size < matrix.shape[0] else float(positive[0] / positive[-1])
    return {
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors_columns": eigenvectors.tolist(),
        "numerical_rank": int(positive.size),
        "condition_number": condition,
    }


def correlation(matrix: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    denominator = np.outer(scale, scale)
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0.0)


def main() -> None:
    args = parse_args()
    blocks = [json.loads((path / "modis_parameter_identifiability.json").read_text()) for path in args.inputs]
    reference = blocks[0]
    controls = reference["parameter_controls"]
    sample_days = reference["sample_days_zero_based"]
    for block in blocks[1:]:
        if block["parameter_controls"] != controls or block["sample_days_zero_based"] != sample_days:
            raise ValueError("block controls or sample days do not match")

    intervals = sorted((block["column_start"], block["column_stop"]) for block in blocks)
    expected_start = 0
    for start, stop in intervals:
        if start != expected_start or stop <= start:
            raise ValueError(f"column blocks are not contiguous at {expected_start}: {(start, stop)}")
        expected_start = stop
    if expected_start != reference["captured_ncolumns"]:
        raise ValueError(f"column blocks stop at {expected_start}, expected {reference['captured_ncolumns']}")

    nparameter = len(controls)
    results = []
    total = np.zeros((nparameter, nparameter), dtype=np.float64)
    calibration = np.zeros_like(total)
    validation = np.zeros_like(total)
    for sample_number, day in enumerate(sample_days):
        periods = {}
        day_information = np.zeros_like(total)
        for period in ("day", "night"):
            entries = [block["results"][sample_number]["periods"][period] for block in blocks]
            weight = sum(entry["area_weight_sum"] for entry in entries)
            samples = sum(entry["samples"] for entry in entries)
            raw_information = sum(
                (np.asarray(entry["information_raw"], dtype=np.float64) for entry in entries),
                start=np.zeros_like(total),
            )
            information = raw_information / weight
            bias = sum(entry["bias_K"] * entry["area_weight_sum"] for entry in entries) / weight
            rmse = np.sqrt(
                sum(entry["rmse_K"] ** 2 * entry["area_weight_sum"] for entry in entries) / weight
            )
            loss_gradient = sum(
                np.asarray(entry["loss_gradient_log_scale"]) * entry["area_weight_sum"]
                for entry in entries
            ) / weight
            periods[period] = {
                "samples": samples,
                "area_weight_sum": weight,
                "bias_K": float(bias),
                "rmse_K": float(rmse),
                "loss_gradient_log_scale": loss_gradient.tolist(),
                "information": information.tolist(),
                "information_raw": raw_information.tolist(),
            }
            day_information += information
        results.append({
            "observation_day_index": day,
            "periods": periods,
            "information": day_information.tolist(),
        })
        total += day_information
        if sample_number < len(sample_days) // 2:
            calibration += day_information
        else:
            validation += day_information

    combined = {
        "parameter_controls": controls,
        "captured_ncolumns": reference["captured_ncolumns"],
        "column_blocks": intervals,
        "sample_days_zero_based": sample_days,
        "calibration_days": reference["calibration_days"],
        "validation_days": reference["validation_days"],
        "strict_lst_error_scale_K": reference["strict_lst_error_scale_K"],
        "rademacher_probes_per_period_per_block": reference["rademacher_probes_per_period"],
        "results": results,
        "information_total": total.tolist(),
        "information_calibration": calibration.tolist(),
        "information_validation": validation.tolist(),
        "spectrum_total": spectrum(total),
        "spectrum_calibration": spectrum(calibration),
        "spectrum_validation": spectrum(validation),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "modis_parameter_identifiability_combined.json").write_text(
        json.dumps(combined, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        args.output / "modis_parameter_identifiability_combined.npz",
        parameter_controls=np.asarray(controls),
        sample_days=np.asarray(sample_days),
        information_total=total,
        information_calibration=calibration,
        information_validation=validation,
    )

    labels = ("LAI", r"$r_{s,min}$", r"$z_{0m}$")
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.6), constrained_layout=True)
    for axis, matrix, title in (
        (axes[0, 0], calibration, "Spring-summer information correlation"),
        (axes[0, 1], validation, "Autumn-winter information correlation"),
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

    calibration_eigenvalues = np.asarray(combined["spectrum_calibration"]["eigenvalues"])
    validation_eigenvalues = np.asarray(combined["spectrum_validation"]["eigenvalues"])
    axes[1, 0].semilogy(range(1, 4), calibration_eigenvalues, "o-", label="spring-summer")
    axes[1, 0].semilogy(range(1, 4), validation_eigenvalues, "s-", label="autumn-winter")
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
    figure.savefig(args.output / "modis_parameter_identifiability.png", dpi=300, bbox_inches="tight")
    print(json.dumps({
        "output": str(args.output),
        "column_blocks": intervals,
        "spectrum_calibration": combined["spectrum_calibration"],
        "spectrum_validation": combined["spectrum_validation"],
    }, indent=2))


if __name__ == "__main__":
    main()
