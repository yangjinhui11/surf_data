#!/usr/bin/env python3
"""Replot the two-timescale closure calibration figure from the cached
asym_train_history.json / asym_train_result.json (no re-run).  Panels:
tau_day/tau_night trajectories, calibration-window subset RMSE, and the
SURF vs SURF-Opt full-domain RMSE bars in both windows."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path("/data/yangjinhui/surf_pytorch/surf_paper/experiments/slab_asym_train_20260824")
BOUNDS_HOURS = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
PERIODS = ("day", "night")


def main() -> None:
    history = json.loads((RUN / "asym_train_history.json").read_text())
    result = json.loads((RUN / "asym_train_result.json").read_text())
    prior = result["prior"]
    trained = result["trained"]

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    iterations_axis = [h["iteration"] for h in history]
    tau_day = [h["controls"][0] for h in history]
    tau_night = [h["controls"][1] for h in history]
    axes[0].plot(iterations_axis, tau_day, marker=".", color="tab:orange", label="tau_day")
    axes[0].plot(iterations_axis, tau_night, marker=".", color="tab:purple", label="tau_night")
    if BOUNDS_HOURS is not None and len(BOUNDS_HOURS) == 4:
        for bound, color in ((BOUNDS_HOURS[0], "tab:orange"), (BOUNDS_HOURS[1], "tab:orange"),
                             (BOUNDS_HOURS[2], "tab:purple"), (BOUNDS_HOURS[3], "tab:purple")):
            axes[0].axhline(bound, color=color, linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set(xlabel="iteration", ylabel="tau (h)", title="Relaxation times (log scale)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
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
    for window_index, window in enumerate(("calibration", "validation")):
        for period_index, period in enumerate(PERIODS):
            for config_index, (config_name, payload) in enumerate(
                (("SURF", prior), ("SURF-Opt", trained))
            ):
                value = payload[window]["summary"][period]["rmse_K"]
                labels.append(f"{window[:3]} {period}\n{config_name}")
                positions.append(window_index * 2.6 + period_index * 1.3 + config_index * 0.6)
                heights.append(value if value is not None else 0.0)
                colors.append(f"C{config_index}")
    axes[2].bar(positions, heights, color=colors, width=0.55)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels, fontsize=6, rotation=30, ha="right")
    axes[2].set(ylabel="LST RMSE (K)", title="RMSE: SURF vs SURF-Opt")
    axes[2].grid(alpha=0.3, axis="y")
    figure.suptitle("Two-timescale surface-layer closure calibration (Adam on tau_day, tau_night) "
                    "against MODIS LST", fontsize=11)
    figure.savefig(RUN / "asym_train_closure.png", dpi=260)
    plt.close(figure)
    print(json.dumps({"done": str(RUN / "asym_train_closure.png")}))


if __name__ == "__main__":
    main()
