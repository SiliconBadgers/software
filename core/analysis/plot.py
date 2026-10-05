"""Plot baseline latency and eligible CPU operation shares for one run or recorded package."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from policy import load_experiment, timing_eligible, DEFAULT_EXPERIMENT
from result_io import exists, load_json, output_dir

COLORS = {"cpu": "#275DAD", "metal": "#148271", "cuda": "#6B4C9A"}
GROUPS = ["MLP projections", "Attention/DeltaNet projections", "Vocabulary output head",
          "Attention QK/softmax/AV", "DeltaNet recurrence", "Copies/gather/state movement",
          "Vector/norm/activation/other", "Convolution"]
LABELS = ["MLP matrices", "Attention / DeltaNet matrices", "Vocabulary head", "Attention core",
          "DeltaNet recurrence", "Copies / gather / state", "Vector / norm / activation", "Convolution"]
PALETTE = ["#275DAD", "#5B8BC3", "#D0782B", "#148271", "#61A78B", "#8E73AC", "#B9A6CA", "#CDBB9B"]
MATRIX_GROUPS = GROUPS[:3]


def token_label(n):
    return f"{n // 1024}K" if n % 1024 == 0 and n >= 1024 else str(n)


def run_names(experiment, backend):
    tags = ["baseline"] + list(experiment["policy"]["separate_case_prefixes"].values())
    return [f"{backend}-{tag}" for tag in tags]


def subtitle_for(results_dir, override):
    if override:
        return override
    manifest = results_dir / "run-manifest.json"
    if manifest.is_file():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        return info.get("plot_subtitle") or "host recorded in run-manifest.json"
    return "host not recorded for this data"


def draw(results_dir, out_dir, experiment, subtitle=None):
    baselines = load_json(results_dir / "summary.json")
    has_ops = exists(results_dir / "operation-summary.json")
    ops = load_json(results_dir / "operation-summary.json")["runs"] if has_ops else []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12, 9 if has_ops else 5), facecolor="white")
    if has_ops:
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.55, wspace=0.27)
        ax1, ax2, ax3 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    else:
        grid = fig.add_gridspec(1, 2, wspace=0.27)
        ax1, ax2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    plotted, decode_hi, prefill_lo, prefill_hi, repetitions, decode_steps = [], 0, 1e9, 0, 0, 0
    for backend in ("cpu", "metal", "cuda"):
        data = sorted([r for r in baselines if r["run"] in run_names(experiment, backend)],
                      key=lambda r: r["prompt_tokens"])
        if not data:
            continue
        plotted.append(backend)
        x = np.arange(len(data))
        repetitions = max(repetitions, max(r["repetitions"] for r in data))
        decode_steps = max(decode_steps, max(r["decode_steps_per_rep"] for r in data))
        prefill_lo = min(prefill_lo, min(r["prefill_min_s"] for r in data))
        prefill_hi = max(prefill_hi, max(r["prefill_max_s"] for r in data))
        decode_hi = max(decode_hi, max(r["decode_max_ms"] for r in data))
        for ax, metric, low, high in ((ax1, "prefill_seconds", "prefill_min_s", "prefill_max_s"),
                                      (ax2, "decode_ms", "decode_min_ms", "decode_max_ms")):
            y = np.array([r[metric] for r in data])
            err = np.array([[r[metric] - r[low] for r in data], [r[high] - r[metric] for r in data]])
            label = {"cpu": "CPU", "metal": "Metal", "cuda": "CUDA"}[backend]
            ax.errorbar(x, y, yerr=err, marker="o", capsize=4, lw=2, color=COLORS[backend], label=label)
            offsets = {"cpu": 8, "metal": -18, "cuda": 18}
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, offsets[backend]),
                            textcoords="offset points", ha="center", fontsize=9, color=COLORS[backend])
            ax.set_xticks(x, [f"{r['prompt_tokens']:,}" for r in data])
            ax.set_xlabel(f"Prompt tokens before {decode_steps} decode steps")
            ax.grid(axis="y", alpha=.16)
    if not plotted:
        raise ValueError("No baseline measurements found to plot")
    ax1.set_yscale("log")
    ax1.set_ylim(prefill_lo / 3, prefill_hi * 2.1)
    ax1.set_ylabel("Prefill seconds · log scale")
    ax1.set_title("Prompt processing", loc="left", fontweight="bold")
    ax2.set_ylim(0, decode_hi * 1.25)
    ax2.set_ylabel("Decode ms / token")
    ax2.set_title("Cached single-token processing", loc="left", fontweight="bold")
    ax1.legend(frameon=False)
    plural = lambda n, word: f"{n} {word}" + ("" if n == 1 else "s")
    notes = [f"Top: uninstrumented medians of {plural(repetitions, 'run')}; whiskers show min–max."]
    if has_ops:
        report_n = experiment["policy"]["report_prompt_tokens"]
        eligible = sorted({r["prompt_tokens"] for r in ops if r.get("timing_eligible_for_report", True)})
        if report_n not in eligible:
            report_n = eligible[-1]
        left, shares = np.zeros(2), {}
        for g, label, c in zip(GROUPS, LABELS, PALETTE):
            vals = []
            for phase in ("prefill", "decode"):
                rows = [r for r in ops if r["prompt_tokens"] == report_n and r["phase"] == phase
                        and r.get("timing_eligible_for_report", True)]
                vals.append(100 * sum(r["groups_us"].get(g, 0) for r in rows) / sum(r["timed_interval_us"] for r in rows))
            shares[g] = vals
            ax3.barh([1, 0], vals, left=left, color=c, label=label, height=.45)
            for i, v in enumerate(vals):
                if v >= 6:
                    ax3.text(left[i] + v / 2, 1 - i, f"{v:.0f}%", ha="center", va="center",
                             color="white" if c not in ["#B9A6CA", "#CDBB9B"] else "#222",
                             fontsize=10, fontweight="bold")
            left += vals
        tag = token_label(report_n)
        matrix = [sum(shares[g][i] for g in MATRIX_GROUPS) for i in range(2)]
        ax3.set_yticks([1, 0], [f"{tag} prefill", f"{tag} decode"])
        ax3.set_xlim(0, 100)
        ax3.set_ylim(-.6, 1.6)
        ax3.set_xlabel("Share of timed CPU operation intervals (%)")
        ax3.set_title(f"CPU work at {tag}: matrix operations are {matrix[0]:.0f}% of prefill and {matrix[1]:.0f}% of decode",
                      loc="left", fontweight="bold", pad=16)
        ax3.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.28), fontsize=9, columnspacing=1.5)
        trace_reps = max(r["rep"] for r in ops) + 1
        notes.append(f"Bottom: {plural(trace_reps, 'CPU trace')}, {plural(decode_steps, 'decode step')} each.")
        notes.append("CPU operation shares do not describe GPU kernels or predict accelerator area."
                     + (" Anomalous timing shares for excluded cases omitted."
                        if any(not timing_eligible(experiment, r["prompt_tokens"]) for r in ops) else ""))
    fig.suptitle("Qwen3.5-2B in llama.cpp", x=.08, y=.98, ha="left", fontsize=22, fontweight="bold")
    fig.text(.08, .935, subtitle or "", fontsize=11, color="#555")
    fig.text(.08, .028, " ".join(notes[:2]) + "\n" + " ".join(notes[2:]), fontsize=9, color="#555")
    fig.subplots_adjust(top=.86, bottom=.21 if has_ops else .18, left=.1, right=.97)
    fig.savefig(out_dir / "profiling-summary.png", dpi=180)
    fig.savefig(out_dir / "profiling-summary.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory holding summary.json and (optionally) operation-summary.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--subtitle", help="Host/config line; default comes from run-manifest.json")
    args = parser.parse_args(argv)
    results_dir = args.results_dir.resolve()
    out = output_dir(args.output_dir)
    draw(results_dir, out, load_experiment(args.experiment), subtitle_for(results_dir, args.subtitle))
    print("Output directory:", out)


if __name__ == "__main__":
    main()
