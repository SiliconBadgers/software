#!/usr/bin/env python3
"""Plot the bounded joint matrix/L1/HBM analytical tradeoff study."""

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
    matrices = document["dimensions"]["matrixCount"]
    banks = document["dimensions"]["l1Banks"]
    bandwidths = document["dimensions"]["hbmGBs"]
    max_hbm = max(bandwidths)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, phase in zip(axes[0], ("prefill", "decode")):
        values = np.array([
            [next(row[phase]["latencyMs"] for row in rows if row["config"] == {"matrixCount": matrix, "l1Banks": bank, "hbmGBs": max_hbm}) for matrix in matrices]
            for bank in banks
        ])
        image = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(matrices)), matrices)
        ax.set_yticks(range(len(banks)), banks)
        ax.set_xlabel("Matrix units")
        ax.set_ylabel("Modeled L1 banks")
        ax.set_title(f"{phase.title()} latency at {max_hbm} GB/s HBM (ms)")
        for y in range(len(banks)):
            for x in range(len(matrices)):
                ax.text(x, y, f"{values[y, x]:.1f}", ha="center", va="center", fontsize=7, color="white" if values[y, x] > np.median(values) else "black")
        fig.colorbar(image, ax=ax, shrink=0.85)

    scatter = axes[1, 0]
    points = scatter.scatter(
        [row["prefill"]["latencyMs"] for row in rows],
        [row["decode"]["latencyMs"] for row in rows],
        c=[row["config"]["hbmGBs"] for row in rows],
        s=[18 + row["config"]["l1Banks"] / 4 for row in rows],
        cmap="plasma",
        alpha=0.7,
    )
    scatter.set_xlabel("Prefill scheduled latency (ms)")
    scatter.set_ylabel("Decode scheduled latency (ms)")
    scatter.set_title("All feasible configurations\ncolor = HBM GB/s; size = modeled L1 banks")
    scatter.grid(True, alpha=0.25)
    fig.colorbar(points, ax=scatter, label="HBM GB/s")

    bank_ax = axes[1, 1]
    prefill, decode = [], []
    for bank in banks:
        row = next(row for row in rows if row["config"] == {"matrixCount": max(matrices), "l1Banks": bank, "hbmGBs": max_hbm})
        prefill.append(row["prefill"]["latencyMs"])
        decode.append(row["decode"]["latencyMs"])
    bank_ax.plot(banks, prefill, marker="o", label="Prefill ms")
    bank_ax.set_xlabel("Modeled L1 banks")
    bank_ax.set_ylabel("Prefill latency (ms)")
    decode_ax = bank_ax.twinx()
    decode_ax.plot(banks, decode, marker="s", color="#D55E00", label="Decode ms")
    decode_ax.set_ylabel("Decode latency (ms)")
    bank_ax.set_title(f"Bank response at {max(matrices)} matrix units, {max_hbm} GB/s")
    bank_ax.grid(True, alpha=0.25)
    lines = bank_ax.lines + decode_ax.lines
    bank_ax.legend(lines, [line.get_label() for line in lines], loc="upper right")

    fig.suptitle("Qwen3.5-2B bounded joint resource tradeoffs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"PASS wrote {args.output}")


if __name__ == "__main__":
    main()
