#!/usr/bin/env python3
"""Plot reproducible CPU/GPU CMFD SURF scaling results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None, help="legacy single-run CSV input")
    parser.add_argument("--summary", type=Path, default=None, help="raw benchmark JSON with repeat ranges")
    parser.add_argument("--output", type=Path, default=root / "surf_cpu_gpu_scaling.png")
    parser.add_argument("--steps", type=int, default=None, help="required when --summary is not supplied")
    return parser.parse_args()


def from_summary(path: Path) -> tuple[np.ndarray, int, dict[tuple[str, int], dict[str, float]]]:
    report = json.loads(path.read_text())
    steps = int(report["configuration"]["steps"])
    columns = np.asarray(report["configuration"]["columns"], dtype=int)
    rows: dict[tuple[str, int], dict[str, float]] = {}
    for row in report["median_results"]:
        rows[(str(row["mode"]), int(row["columns"]))] = {
            "median": float(row["integration_seconds_median"]),
            "minimum": float(row["integration_seconds_min"]),
            "maximum": float(row["integration_seconds_max"]),
            "setup": float(row["setup_seconds_median"]),
        }
    return columns, steps, rows


def from_csv(path: Path, steps: int) -> tuple[np.ndarray, int, dict[tuple[str, int], dict[str, float]]]:
    with path.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    columns = np.asarray([int(row["columns"]) for row in source])
    rows: dict[tuple[str, int], dict[str, float]] = {}
    for row in source:
        ncol = int(row["columns"])
        cpu = float(row["cpu_integration_seconds"])
        gpu = float(row["gpu_preload_integration_seconds"])
        rows[("cpu", ncol)] = {"median": cpu, "minimum": cpu, "maximum": cpu, "setup": 0.0}
        rows[("gpu", ncol)] = {
            "median": gpu,
            "minimum": gpu,
            "maximum": gpu,
            "setup": float(row.get("gpu_preload_setup_seconds", 0.0)),
        }
    return columns, steps, rows


def throughput_errors(columns: np.ndarray, steps: int, values: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    median = np.asarray([value["median"] for value in values])
    minimum = np.asarray([value["minimum"] for value in values])
    maximum = np.asarray([value["maximum"] for value in values])
    rate = columns * steps / median
    lower = rate - columns * steps / maximum
    upper = columns * steps / minimum - rate
    return rate, np.vstack((lower, upper))


def main() -> None:
    args = parse_args()
    if args.summary is not None:
        columns, steps, records = from_summary(args.summary)
    elif args.results is not None:
        if args.steps is None:
            raise SystemExit("--steps is required when --summary is not supplied")
        columns, steps, records = from_csv(args.results, args.steps)
    else:
        raise SystemExit("supply --summary from benchmark_scaling.py or --results with --steps")
    if np.any(columns < 1) or steps < 1:
        raise ValueError("columns and steps must be positive")

    cpu = [records[("cpu", int(ncol))] for ncol in columns]
    gpu = [records[("gpu", int(ncol))] for ncol in columns]
    cpu_rate, cpu_error = throughput_errors(columns, steps, cpu)
    gpu_rate, gpu_error = throughput_errors(columns, steps, gpu)
    cpu_seconds = np.asarray([item["median"] for item in cpu])
    gpu_seconds = np.asarray([item["median"] for item in gpu])
    speedup = cpu_seconds / gpu_seconds

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11})
    figure, (ax_rate, ax_speedup) = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    ax_rate.errorbar(
        columns, cpu_rate, yerr=cpu_error, fmt="o-", color="#ba5b30", lw=2,
        capsize=3, label="CPU (32 threads, median and range)",
    )
    ax_rate.errorbar(
        columns, gpu_rate, yerr=gpu_error, fmt="s-", color="#1976a2", lw=2,
        capsize=3, label="A100 GPU (median and range)",
    )
    ax_rate.set_xscale("log")
    ax_rate.set_yscale("log")
    ax_rate.set_xlabel("Number of active land columns")
    ax_rate.set_ylabel("Integration throughput (column-steps s$^{-1}$)")
    ax_rate.set_title(f"{steps}-step integration throughput")
    ax_rate.grid(True, which="both", alpha=0.28)
    ax_rate.legend(frameon=False, loc="upper left", fontsize=8)

    ax_speedup.plot(columns, speedup, "o-", color="#147a5b", lw=2)
    ax_speedup.axhline(1.0, color="0.35", lw=1, ls="--")
    ax_speedup.set_xscale("log")
    ax_speedup.set_xlabel("Number of active land columns")
    ax_speedup.set_ylabel("GPU / CPU integration speed-up")
    ax_speedup.set_title("GPU advantage increases with batch size")
    ax_speedup.set_ylim(0.0, float(np.max(speedup)) * 1.16)
    ax_speedup.grid(True, which="both", alpha=0.28)
    for x, y in zip(columns, speedup):
        ax_speedup.annotate(f"{y:.1f}x", (x, y), xytext=(0, 7), textcoords="offset points", ha="center")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    for ncol, cpu_item, gpu_item, factor in zip(columns, cpu, gpu, speedup):
        print(
            f"{ncol:6d}: CPU={cpu_item['median']:.3f} s "
            f"[{cpu_item['minimum']:.3f}, {cpu_item['maximum']:.3f}], "
            f"GPU={gpu_item['median']:.3f} s "
            f"[{gpu_item['minimum']:.3f}, {gpu_item['maximum']:.3f}], "
            f"GPU setup={gpu_item['setup']:.3f} s, speed-up={factor:.2f}x"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
