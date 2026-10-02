"""Decode time versus context length: how the cost of a continuation step changes as the KV cache grows.

The timing summaries average every decode step of a repetition, which hides that trend. A trend is only
visible when the continuation spans a real fraction of the context (context_span_fraction) and the fit
(fit_r_squared) is good; over 32 steps of a 2,048-token prompt the span is under 2% and any slope is noise. Here each step keeps
its own value (the median over repetitions) and is placed at its context length: step j of a prompt of n tokens
attends over n + j + 1 tokens. Untraced runs only; instrumented runs carry tracing overhead.
"""
import csv
import json
import statistics

from result_io import exists, open_csv, open_result, write_json

MAX_BINS = 16
MAX_WINDOW = 8


def _measurements(path):
    with open_result(path) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [r for r in rows if r["kind"] == "measurement"]


def least_squares(xs, ys):
    """(slope, intercept, r_squared); None values when fewer than 3 points or no spread in x."""
    n = len(xs)
    if n < 3:
        return None, None, None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None, None, None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    syy = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 if syy == 0 else (sxy * sxy) / (sxx * syy)
    return slope, intercept, r2


def bins(context, per_step, count=MAX_BINS):
    """Contiguous, near-equal groups of steps: [{first_context, last_context, steps, mean_ms}]."""
    count = max(1, min(count, len(per_step)))
    out, start = [], 0
    for i in range(count):
        stop = start + len(per_step) // count + (1 if i < len(per_step) % count else 0)
        out.append({"first_context": context[start], "last_context": context[stop - 1], "steps": stop - start,
                    "mean_ms": statistics.mean(per_step[start:stop])})
        start = stop
    return out


def curve(measurements):
    """One run at one prompt length -> per-step medians and summary statistics. Repetitions must agree on shape."""
    n = measurements[0]["prompt_tokens"]
    steps = len(measurements[0]["decode_us"])
    if any(len(m["decode_us"]) != steps for m in measurements):
        raise ValueError("Repetitions disagree on the number of decode steps")
    per_step = [statistics.median(m["decode_us"][j] for m in measurements) / 1000 for j in range(steps)]
    context = [n + j + 1 for j in range(steps)]
    window = max(1, min(MAX_WINDOW, steps // 4))
    first, last = statistics.mean(per_step[:window]), statistics.mean(per_step[-window:])
    slope, intercept, r2 = least_squares(context, per_step)
    return {
        "prompt_tokens": n, "repetitions": len(measurements), "steps": steps,
        "first_context_tokens": context[0], "last_context_tokens": context[-1],
        "context_span_fraction": (context[-1] - context[0]) / context[0],
        "window_steps": window, "first_window_mean_ms": first, "last_window_mean_ms": last,
        "growth_ratio": last / first if first else None,
        "slope_ms_per_1000_context_tokens": None if slope is None else 1000 * slope,
        "fit_intercept_ms": intercept, "fit_r_squared": r2,
        "median_ms": statistics.median(per_step), "min_ms": min(per_step), "max_ms": max(per_step),
        "bins": bins(context, per_step), "context_tokens": context, "per_step_ms": per_step}


def compact(c):
    """The curve without the full per-step arrays, for embedding in other JSON."""
    return {k: v for k, v in c.items() if k not in ("context_tokens", "per_step_ms")}


def curves_for_file(path):
    """{prompt_tokens: curve} for every prompt length measured in one .jsonl file."""
    by_length = {}
    for m in _measurements(path):
        by_length.setdefault(m["prompt_tokens"], []).append(m)
    return {n: curve(rows) for n, rows in sorted(by_length.items())}


def summarize_run(results_dir, experiment):
    """Curves for every untraced measurement file listed in the experiment's summary_runs."""
    found = []
    for name in experiment["policy"]["summary_runs"]:
        if "profile" in name:              # instrumented runs: tracing overhead is inside their timings
            continue
        path = results_dir / (name + ".jsonl")
        if not exists(path):
            continue
        for n, c in curves_for_file(path).items():
            found.append({"run": name, **c})
    return found


def write_outputs(out_dir, curves):
    write_json(out_dir / "decode-curve.json", {
        "note": "Per-step decode time (median over repetitions) at each context length; untraced runs only.",
        "curves": curves})
    with open_csv(out_dir / "decode-curve.csv") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "prompt_tokens", "step", "context_tokens", "decode_ms_median"])
        for c in curves:
            for j, (ctx, ms) in enumerate(zip(c["context_tokens"], c["per_step_ms"])):
                writer.writerow([c["run"], c["prompt_tokens"], j, ctx, ms])


def plot(out_dir, curves):
    """decode-curve.png: one line per (run, prompt length). Skipped when no run has more than one step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    usable = [c for c in curves if c["steps"] > 1]
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    for c in usable:
        ax.plot(c["context_tokens"], c["per_step_ms"], lw=1.4,
                label=f"{c['run']} · prompt {c['prompt_tokens']:,} ({c['steps']} steps)")
    ax.set_xlabel("Tokens in the KV cache when the step runs")
    ax.set_ylabel("Decode ms per token (median over repetitions)")
    ax.set_title("Decode cost as the context grows", loc="left", fontweight="bold")
    ax.grid(alpha=.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "decode-curve.png", dpi=160)
    plt.close(fig)
    return out_dir / "decode-curve.png"
