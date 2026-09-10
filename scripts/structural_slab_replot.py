#!/usr/bin/env python3
"""Replot the structural-closure trajectory figure (slab mode) from the
cached structural_history.json / structural_result.json, with the renamed
"surface-layer closure" terminology (no re-run of the calibration)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = Path("/data/yangjinhui/surf_pytorch/surf_paper/experiments/structural_closure_20260823/slab")
PERIODS = ("day", "night")


def main() -> None:
    history = json.loads((RUN / "structural_history.json").read_text())
    result = json.loads((RUN / "structural_result.json").read_text())
    baseline_full = result["baseline_full_domain"]
    optimized_full = result["optimized_full_domain"]

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    iterations_axis = [h["iteration"] for h in history]
    axes[0].plot(iterations_axis, [h["controls"][0] for h in history], marker=".")
    axes[0].set(xlabel="iteration", ylabel="lambda_s  K/(W m-2)",
                title="Coupling coefficient")
    axes[0].grid(alpha=0.3)
    for period, color in zip(PERIODS, ("tab:red", "tab:blue")):
        values = []
        for h in history:
            entries = [d[period]["rmse_K"] for d in h["days"].values() if d[period]["rmse_K"] is not None]
            values.append(float(np.sqrt(np.mean(np.square(entries)))) if entries else float("nan"))
        axes[1].plot(iterations_axis, values, color=color, label=period)
    axes[1].set(xlabel="iteration", ylabel="LST RMSE (K)", title="Calibration window (training subset)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    labels, positions, heights, colors = [], [], [], []
    for window_index, window in enumerate(("calibration_window", "validation_window")):
        for period_index, period in enumerate(PERIODS):
            for config_index, (config_name, payload) in enumerate(
                (("SURF", baseline_full), ("SURF-Opt", optimized_full))
            ):
                value = payload[window][period]["rmse_K"]
                labels.append(f"{window.split('_')[0][:3]} {period}\n{config_name}")
                positions.append(window_index * 2.6 + period_index * 1.3 + config_index * 0.6)
                heights.append(value if value is not None else 0.0)
                colors.append(f"C{config_index}")
    axes[2].bar(positions, heights, color=colors, width=0.55)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels, fontsize=6, rotation=30, ha="right")
    axes[2].set(ylabel="LST RMSE (K)", title="RMSE: SURF vs SURF-Opt")
    axes[2].grid(alpha=0.3, axis="y")
    figure.suptitle("Structural correction experiment (one-parameter surface-layer closure) "
                    "against MODIS LST", fontsize=11)
    figure.savefig(RUN / "structural_closure.png", dpi=260)
    plt.close(figure)
    print(json.dumps({"done": str(RUN / "structural_closure.png")}))


if __name__ == "__main__":
    main()
