#!/usr/bin/env python3
"""Plot class-wide recurrence multiplier sensitivity results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [row for row in document["rows"] if row["valid"]]
    penalties = document["dimensions"]["penalties"]
    matrices = document["dimensions"]["matrixCounts"]
    reference = [
        row for row in rows
        if row["workload"] == "target-batch1"
        and row["precision"] == "w8a16"
        and row["memoryProfile"] == "f2-ceiling"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, phase in zip(axes[0], ("prefill", "decode")):
        for color, matrix_count in zip(("#0072B2", "#D55E00", "#009E73"), (4, 8, 16)):
            values = [next(
                row[phase]["latencyMs"] for row in reference
                if row["mapping"] == "matrix"
                and row["matrixCount"] == matrix_count
                and row["recurrentMatrixPenalty"] == penalty
            ) for penalty in penalties]
            dedicated = next(
                row[phase]["latencyMs"] for row in reference
                if row["mapping"] == "dedicated"
                and row["matrixCount"] == matrix_count
                and row["recurrentCount"] == 4
            )
            ax.plot(penalties, values, marker="o", color=color, label=f"shared matrix, M={matrix_count}")
            ax.axhline(dedicated, color=color, linestyle="--", alpha=0.75, label=f"dedicated R=4, M={matrix_count}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(penalties, penalties)
        ax.set_xlabel("32-bit recurrence penalty (W8/A16-equivalent cycles)")
        ax.set_ylabel(f"{phase.title()} scheduled latency (ms, log scale)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    for ax, phase in zip(axes[1], ("prefill", "decode")):
        values = np.array([
            [
                next(item for item in document["summary"]["penaltyComparisons"] if item["penalty"] == penalty)["phases"][phase]["byMatrixCount"][str(matrix_count)]["beatsVector"]
                for matrix_count in matrices
            ]
            for penalty in penalties
        ])
        totals = np.array([
            [
                next(item for item in document["summary"]["penaltyComparisons"] if item["penalty"] == penalty)["phases"][phase]["byMatrixCount"][str(matrix_count)]["total"]
                for matrix_count in matrices
            ]
            for penalty in penalties
        ])
        percentages = 100 * values / totals
        image = ax.imshow(percentages, aspect="auto", origin="lower", vmin=0, vmax=100, cmap="viridis")
        ax.set_xticks(range(len(matrices)), matrices)
        ax.set_yticks(range(len(penalties)), penalties)
        ax.set_xlabel("Matrix units")
        ax.set_ylabel("32-bit recurrence penalty")
        ax.set_title(f"{phase.title()}: shared matrix beats vector across classes")
        for y in range(len(penalties)):
            for x in range(len(matrices)):
                ax.text(x, y, f"{values[y, x]}/{totals[y, x]}", ha="center", va="center", fontsize=8, color="white" if percentages[y, x] < 45 else "black")
        fig.colorbar(image, ax=ax, label="Class cells won (%)", shrink=0.85)

    fig.suptitle("Qwen3.5-2B F32 gated-delta recurrence mapping sensitivity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"PASS wrote {args.output}")


if __name__ == "__main__":
    main()
