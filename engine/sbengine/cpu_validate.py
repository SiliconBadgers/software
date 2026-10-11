"""Validate the engine's work/traffic accounting against measured CPU op timings.

What this tests: that the MAC counts, byte counts and shape features the engine derives from a captured graph
explain measured per-node CPU time, including for prompt lengths the fit never saw. Per op class we fit
    seconds = a * work + b * bytes + c        (non-negative least squares)
on the training graphs (pp128, pp512) and predict every held-out per-prompt graph (71..628 tokens).

What this does NOT test: the accelerator timing equations (matrix array cycles, L1/HBM service, scheduling).
CPU speed says nothing about those; that needs RTL or synthesis measurements from the Compute team.
Timings come from an instrumented run (profiler overhead is measured separately) and are one run per graph.
"""
import re

import numpy as np

from .graph import has_capture, load_workload
from .trace import load_trace


def node_features(g, n):
    """(class, {feature: value}) for one node. Same accounting as the engine, no hardware config involved."""
    t, op = n.tensor, n.op
    if op == "MUL_MAT":
        a = n.slots[0]
        macs = float(t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3] * a.shape[0])
        cls = f"matmul[{a.dtype}]" if g.kind(g.root(a.id)) == "weight" else "matmul[act]"
        return cls, {"macs": macs, "bytes": float(a.nbytes), "one": 1.0}
    if op == "FLASH_ATTN_EXT":
        q, k, v = n.slots[0], n.slots[1], n.slots[2]
        macs = 2.0 * q.shape[0] * q.shape[1] * q.shape[2] * k.shape[1]
        return "flash_attn", {"macs": macs, "bytes": float(k.nbytes + v.nbytes), "one": 1.0}
    if op == "GATED_DELTA_NET":
        v = n.slots[2]
        return "gated_delta_net", {"macs": float(v.shape[0] ** 2 * v.shape[1] * g.meta["input_tokens"]), "one": 1.0}
    if op == "SSM_CONV":
        return "ssm_conv", {"macs": float(t.elements * n.slots[1].shape[0]), "one": 1.0}
    if op in ("GET_ROWS", "SET_ROWS", "CPY", "CONT", "CONCAT"):
        return op.lower(), {"bytes": float(t.nbytes), "one": 1.0}
    return op.lower(), {"elems": float(t.elements), "one": 1.0}


def nnls(A, b, tol=1e-12):
    """Lawson-Hanson non-negative least squares (numpy only)."""
    m, n = A.shape
    x, passive = np.zeros(n), []
    for _ in range(3 * n + 10):
        w = A.T @ (b - A @ x)
        rest = [j for j in range(n) if j not in passive]
        if not rest:
            break
        j = max(rest, key=lambda i: w[i])
        if w[j] <= tol:
            break
        passive.append(j)
        while True:
            s = np.zeros(n)
            s[passive] = np.linalg.lstsq(A[:, passive], b, rcond=None)[0]
            if all(s[i] > tol for i in passive):
                x = s
                break
            alpha = min(x[i] / (x[i] - s[i]) for i in passive if s[i] <= tol)
            x = x + alpha * (s - x)
            passive = [i for i in passive if x[i] > tol]
    return x


def collect(run_dir, names, not_checked=None):
    """Samples per (workload, phase): list of (class, features, measured seconds). Phases without a comparable
    trace (e.g. a prompt split into several micro-batches) are skipped. Everything skipped is appended to
    `not_checked` with its reason, so a run without its local-only traces is never mistaken for a checked one."""
    data = {}
    not_checked = [] if not_checked is None else not_checked
    for name in names:
        directory = run_dir / "graphs" / name
        if not has_capture(directory):
            not_checked.append({"workload": name, "phase": "prefill and decode", "reason": "no captured graph"})
            continue
        w = load_workload(directory)
        trace = load_trace(run_dir, w)
        if trace is None:
            not_checked.append({"workload": name, "phase": "prefill and decode", "reason": "no operation trace"})
            continue
        for phase in ("prefill", "decode"):
            if not trace.get(phase):
                not_checked.append({"workload": name, "phase": phase, "reason": "no comparable trace for this phase"})
                continue
            g = w.graph(phase)
            samples = []
            for n in g.nodes:
                sec = trace[phase].get(n.spos)
                if sec is None:
                    continue
                cls, feats = node_features(g, n)
                samples.append((cls, feats, sec))
            meta_time = sum(trace[phase].get(sp, 0.0) for sp in range(len(g.order))) - sum(s[2] for s in samples)
            data[(name, phase)] = {"samples": samples, "metadata_seconds": meta_time, "tokens": w.meta["prompt_tokens"]}
    return data


def fit(train):
    """One non-negative linear model per (phase, op class): CPU kernels differ between the GEMM-like prefill regime and
    the GEMV/bandwidth-bound decode regime, and prefill's long times would otherwise dominate a joint fit."""
    by_class = {}
    for (_, phase), entry in train.items():
        for cls, feats, sec in entry["samples"]:
            by_class.setdefault((phase, cls), []).append((feats, sec))
    models = {}
    for key, rows in by_class.items():
        keys = sorted({k for f, _ in rows for k in f})
        A = np.array([[f.get(k, 0.0) for k in keys] for f, _ in rows])
        b = np.array([sec for _, sec in rows])
        scale = np.maximum(np.abs(A).max(axis=0), 1e-30)
        models[key] = dict(zip(keys, (nnls(A / scale, b) / scale).tolist()))
    return models


def predict(models, phase, cls, feats):
    m = models.get((phase, cls))
    return 0.0 if m is None else sum(m.get(k, 0.0) * v for k, v in feats.items())


def evaluate_heldout(models, data):
    rows = []
    for (name, phase), entry in sorted(data.items()):
        measured, predicted, by_class = 0.0, 0.0, {}
        for cls, feats, sec in entry["samples"]:
            p = predict(models, phase, cls, feats)
            measured += sec
            predicted += p
            c = by_class.setdefault(cls, [0.0, 0.0])
            c[0] += sec
            c[1] += p
        worst = max((abs(100 * (v[1] - v[0]) / measured) for v in by_class.values()), default=0.0)
        rows.append({"workload": name, "phase": phase, "tokens": entry["tokens"], "measured_s": measured,
                     "predicted_s": predicted, "error_pct": 100 * (predicted / measured - 1) if measured else 0.0,
                     "worst_class_error_pp": worst, "unattributed_metadata_s": entry["metadata_seconds"],
                     "by_class": {k: {"measured_s": v[0], "predicted_s": v[1]} for k, v in by_class.items()}})
    return rows


def run(run_dir, train_names=("pp128-fa-on", "pp512-fa-on")):
    """`checked` is True only when at least one held-out graph was predicted from a fit on the training traces.
    Without the training traces nothing is predicted (a fit on no data would predict zero for everything)."""
    prompts = sorted(p.name for p in (run_dir / "graphs").glob("prompt-*-fa-on"))
    not_checked = []
    train = collect(run_dir, list(train_names), not_checked)
    test = collect(run_dir, prompts, not_checked)
    models = fit(train)
    heldout = evaluate_heldout(models, test) if train else []
    if not train:
        reason = "Nothing was fitted: no training graph has a comparable operation trace"
    elif not prompts:
        reason = "No held-out per-prompt graphs in this run"
    elif not heldout:
        reason = "No held-out graph has a comparable operation trace"
    else:
        reason = None
    return {"train": list(train_names), "test": prompts, "models": models, "checked": bool(heldout),
            "not_checked_reason": reason, "not_checked": not_checked,
            "train_fit": evaluate_heldout(models, train), "heldout": heldout}


def report_markdown(res, run_name, top=8):
    """A small Markdown report of run(): whole-graph and per-operation-class measured and predicted CPU time."""
    def ms(x):
        return f"{x * 1e3:,.1f}"

    def class_table(rows, limit=None):
        total = {}
        for r in rows:
            for cls, v in r["by_class"].items():
                t = total.setdefault(cls, [0.0, 0.0])
                t[0] += v["measured_s"]
                t[1] += v["predicted_s"]
        measured_all = sum(v[0] for v in total.values())
        ordered = sorted(total.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0]))
        shown = ordered if limit is None else ordered[:limit]
        out = ["| Operation class | Measured, ms | Predicted, ms | Difference, ms | Difference, % of all measured |",
               "|---|---:|---:|---:|---:|"]
        for cls, (m, p) in shown:
            out.append(f"| `{cls}` | {ms(m)} | {ms(p)} | {(p - m) * 1e3:+,.1f} | {100 * (p - m) / measured_all:+.1f}% |")
        if len(shown) < len(ordered):
            rest_m, rest_p = sum(v[0] for _, v in ordered[limit:]), sum(v[1] for _, v in ordered[limit:])
            out.append(f"| {len(ordered) - len(shown)} other classes | {ms(rest_m)} | {ms(rest_p)} | {(rest_p - rest_m) * 1e3:+,.1f} | "
                       f"{100 * (rest_p - rest_m) / measured_all:+.1f}% |")
        predicted_all = sum(v[1] for v in total.values())
        out.append(f"| **All** | **{ms(measured_all)}** | **{ms(predicted_all)}** | **{(predicted_all - measured_all) * 1e3:+,.1f}** | "
                   f"**{100 * (predicted_all / measured_all - 1):+.1f}%** |")
        return out

    skipped = res.get("not_checked", [])
    lines = [f"# CPU accounting check: {run_name}", "",
             f"Generated from `engine/` by `python -m sbengine validate-cpu --run ../results/{run_name} --report <path>` "
             "(`engine/sbengine/cpu_validate.py`). It needs the run's local-only operation traces and captured graphs.", ""]
    if not res.get("checked", bool(res["heldout"])):
        lines += [f"**Status: NOT CHECKED.** {res.get('not_checked_reason') or 'No held-out graph was compared'}. "
                  "This report contains no validation result. The operation traces and captured graphs are local-only "
                  "files that are not committed; restore them, or run `profile` again, and regenerate this report.", ""]
        if skipped:
            lines += ["| Workload | Phase | Why it was not checked |", "|---|---|---|"]
            lines += [f"| {s['workload']} | {s['phase']} | {s['reason']} |" for s in skipped]
            lines.append("")
        return "\n".join(lines)
    left_out = (f"; {len(skipped)} workload phase{' was' if len(skipped) == 1 else 's were'} not checked and "
                f"{'is' if len(skipped) == 1 else 'are'} listed at the end." if skipped else ".")
    lines += [f"**Status: checked.** {len(res['heldout'])} held-out graph phases were compared{left_out}", "",
              "Per operation class and phase, `seconds = a * work + b * bytes + c` is fitted (non-negative least squares) on "
             f"the training graphs ({', '.join(res['train'])}) and used to predict every held-out per-prompt graph. This "
             "checks that the engine's work and byte counts explain measured CPU time on this host. It does not check the "
             "accelerator timing equations, and the timings are one instrumented run per graph.", "",
             "## Whole graph", "",
             "| Set | Workload | Phase | Tokens | Measured, ms | Predicted, ms | Error |", "|---|---|---|---:|---:|---:|---:|"]
    for label, rows in (("training", res["train_fit"]), ("held out", res["heldout"])):
        for r in rows:
            lines.append(f"| {label} | {r['workload']} | {r['phase']} | {r['tokens']} | {ms(r['measured_s'])} | {ms(r['predicted_s'])} | "
                         f"{r['error_pct']:+.1f}% |")
    lines.append("")
    for phase in ("prefill", "decode"):
        errors = [abs(r["error_pct"]) for r in res["heldout"] if r["phase"] == phase]
        if errors:
            lines.append(f"Held-out {phase}: mean absolute error {sum(errors) / len(errors):.1f}%, max {max(errors):.1f}% "
                         f"(n = {len(errors)}).")
    lines.append("")
    for phase in ("prefill", "decode"):
        rows = [r for r in res["heldout"] if r["phase"] == phase]
        if rows:
            lines += [f"## Held-out prompts, {phase}, by operation class", "",
                      f"Summed over the {len(rows)} held-out graphs; the {top} classes with the largest difference.", ""]
            lines += class_table(rows, top) + [""]
    for r in res["train_fit"]:
        if r["phase"] == "prefill":
            lines += [f"## Training graph {r['workload']}, prefill, by operation class", ""] + class_table([r], top) + [""]
    lines += ["A phase is left out when its trace is not comparable with the captured graph (for example a prompt the "
              "profiled run split into several micro-batches).", ""]
    if skipped:
        lines += ["## Not checked", "", "| Workload | Phase | Why it was not checked |", "|---|---|---|"]
        lines += [f"| {s['workload']} | {s['phase']} | {s['reason']} |" for s in skipped]
        lines.append("")
    return "\n".join(lines)


def effective_rates(models):
    """Human-readable CPU rates implied by the fit (for the report, not used by the accelerator model)."""
    out = {}
    for (phase, cls), m in models.items():
        row = {}
        if m.get("macs", 0) > 0:
            row["GMAC_per_s"] = 1e-9 / m["macs"]
        if m.get("bytes", 0) > 0:
            row["GB_per_s"] = 1e-9 / m["bytes"]
        if m.get("elems", 0) > 0:
            row["Gelem_per_s"] = 1e-9 / m["elems"]
        if m.get("one", 0) > 0:
            row["fixed_us_per_call"] = m["one"] * 1e6
        out[f"{phase}:{cls}"] = row
    return out
