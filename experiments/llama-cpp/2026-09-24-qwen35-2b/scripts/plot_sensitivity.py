#!/usr/bin/env python3
"""Render static sensitivity curves from sweep_sensitivity.cjs JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plot_sweep(sweep: dict, output_dir: Path, pdf: PdfPages) -> None:
    rows = [row for row in sweep["rows"] if row["valid"]]
    labels = [str(row["value"]) for row in rows]
    x = list(range(len(rows)))
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    colors = {"prefill": "#0072B2", "decode": "#D55E00"}

    service_colors = {"compute": "#009E73", "l1": "#CC79A7", "hbm": "#E69F00"}
    for col, phase in enumerate(("prefill", "decode")):
        axes[0, col].plot(x, [row[phase]["latencyMs"] for row in rows], marker="o", color=colors[phase])
        axes[0, col].set_ylabel(f"{phase.title()} modeled latency (ms)")
        axes[1, col].plot(x, [row[phase]["throughput"] for row in rows], marker="o", color=colors[phase])
        axes[1, col].set_ylabel(f"{phase.title()} modeled throughput (tokens/s)")
        ax = axes[2, col]
        for service in ("compute", "l1", "hbm"):
            ax.plot(x, [row[phase]["serviceMs"][service] for row in rows], marker=".", label=service, color=service_colors[service])
        ax.set_ylabel(f"{phase.title()} aggregate service demand (ms)")

    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.set_xlabel(f'{sweep["label"]} ({sweep.get("unit", "")})'.rstrip())
        ax.grid(True, alpha=0.25)
    for ax in axes[2, :]:
        ax.legend()

    for col, phase in enumerate(("prefill", "decode")):
        knee = sweep["knees"][phase]["value"]
        if knee is not None and knee in [row["value"] for row in rows]:
            idx = [row["value"] for row in rows].index(knee)
            bound = sweep["knees"][phase].get("atLowerSampleBound", False)
            label = "≤2% knee at/below sample" if bound else "2% sustained-doubling knee"
            for ax in axes[:, col]:
                ax.axvline(idx, color="#444444", linestyle="--", linewidth=1)
            axes[0, col].annotate(label, (idx, axes[0, col].get_ylim()[1]), xytext=(4, -15), textcoords="offset points", va="top", fontsize=8)

    fig.suptitle(f'{sweep["label"]}: one-dimensional analytical sensitivity')
    stem = sweep["parameter"].replace("_", "-")
    fig.savefig(output_dir / f"{stem}.png", dpi=180)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output_dir / "sensitivity-curves.pdf") as pdf:
        for sweep in document["sweeps"]:
            plot_sweep(sweep, args.output_dir, pdf)
    print(f'PASS rendered {len(document["sweeps"])} sensitivity figures in {args.output_dir}')


if __name__ == "__main__":
    main()
