"""Design sweeps, Pareto fronts, equation ablation, sensitivity and Monte-Carlo rank stability.

Every output is tied to its full assumptions: sweep files carry the base config (JSON), its hash, the
axes, the workload identity and the engine version, so a CSV can be reproduced or challenged later.
"""
import copy
import csv
import itertools
import json
import math
import random
from pathlib import Path

from . import VERSION
from .config import apply_sets, config_hash, conservative, deep_merge, get_path, set_path, switch_report, validate
from .model import evaluate

BINDING = ("matrix", "vector", "recurrent", "dma", "scalar", "host", "l1", "hbm")


def summarize(result):
    """Flat record of the numbers a design decision needs."""
    p, d = result["phases"]["prefill"], result["phases"]["decode"]

    def binding(phase):
        t = phase["time_by_bound"]
        return max(t, key=t.get) if t else ""
    return {
        "valid": result["valid"], "errors": "; ".join(result["errors"]),
        "cost": result["resources"]["cost"], "dsp": result["resources"]["dsp"],
        "prefill_s": p["seconds"], "decode_s": d["seconds"],
        "prefill_tps": p["tokens_per_s"], "decode_tps_per_seq": d["per_sequence_tps"],
        "decode_tps_total": d["per_sequence_tps"] * result["batch"],
        "prefill_lower_s": p["bounds"]["lower_bound"], "prefill_serial_s": p["bounds"]["serial"],
        "decode_lower_s": d["bounds"]["lower_bound"], "decode_serial_s": d["bounds"]["serial"],
        "prefill_binding": binding(p), "decode_binding": binding(d),
        "decode_l1_service_s": d["l1_service_s"], "decode_hbm_service_s": d["hbm_service_s"],
        "hbm_required_gib": max(p["footprint"]["hbm_required"], d["footprint"]["hbm_required"]) / 2**30,
        "config_hash": result["config_hash"],
    }


def grid(base, axes):
    """axes: {'matrix.count': [1, 2], ...} -> iterator of (assignment, config); unresolved combos are kept and flagged."""
    keys = list(axes)
    for values in itertools.product(*(axes[k] for k in keys)):
        cfg = copy.deepcopy(base)
        assignment = dict(zip(keys, values))
        for k, v in assignment.items():
            set_path(cfg, k, v)
        yield assignment, cfg


def run_sweep(workload, base, axes, trace=None, limit=20000):
    combos = math.prod(len(v) for v in axes.values()) if axes else 1
    if combos > limit:
        raise ValueError(f"sweep has {combos} points; limit is {limit}")
    records = []
    for assignment, cfg in grid(base, axes):
        problems = validate(cfg)
        if problems:
            records.append({**assignment, "valid": False, "errors": "; ".join(problems)})
            continue
        records.append({**assignment, **summarize(evaluate(workload, cfg, trace))})
    return records


def dominates(a, b, objectives):
    better = False
    for key, sense in objectives:
        x, y = a[key], b[key]
        if sense == "max":
            x, y = -x, -y
        if x > y:
            return False
        if x < y:
            better = True
    return better


def pareto(records, objectives=(("cost", "min"), ("decode_s", "min"))):
    """Non-dominated valid records; points with identical objective values are reported once (the first)."""
    valid = [r for r in records if r.get("valid")]
    front, seen = [], set()
    for r in valid:
        key = tuple(r[k] for k, _ in objectives)
        if key in seen or any(dominates(o, r, objectives) for o in valid if o is not r):
            continue
        seen.add(key)
        front.append(r)
    return front


def write_sweep(out_dir, name, records, base, axes, workload, command=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = list(axes) + [k for k in (records[0] if records else {}) if k not in axes]
    with (out_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(columns)
        for r in records:
            w.writerow([r.get(c, "") for c in columns])
    meta = {"engine_version": VERSION, "workload": workload.name, "prompt_tokens": workload.meta["prompt_tokens"],
            "flash_attention": workload.meta.get("flash_attention"), "base_config_hash": config_hash(base),
            "base_config": base, "axes": axes, "points": len(records), "command": command}
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_dir / f"{name}.csv"


# -- ablation ---------------------------------------------------------------------------------------------------
# Each step switches ONE equation from the explorer's version to this engine's. Order matters for the cumulative
# table; the leave-one-out table reverts each step alone from the full engine.
ABLATION_STEPS = [
    ("weights: captured per-tensor formats (was uniform 8-bit)", {"precision": {"weights": "captured"}}),
    ("matmul tiling: searched tile (was fixed tileN=64)", {"matrix": {"tile_mode": "search"}}),
    ("array fill/drain once per op (was once per tile)", {"matrix": {"pipelined_tiles": True}}),
    ("L1 feed: array-level operand streaming (was tile-level)", {"memory": {"l1_feed": "array"}}),
    ("activations: liveness/residency simulation (was per-op spill proxy)", {"memory": {"residency": "belady"}}),
    ("fusion: single-consumer epilogues never stored (was none)", {"memory": {"fusion": "chains"}}),
    ("recurrence: serial or chunked, cheaper wins (was serial only)", {"recurrent": {"lowering": "auto"}}),
    ("attention: causal tile skipping in fused attention (n/a for fa-off)", {"attention": {"causal_skip": True}}),
    ("cross-unit hand-off cost (was none)", {"sync_cycles": 100}),
    ("schedule: dependency-aware list schedule (was serial sum)", {"schedule": {"mode": "pools"}}),
]


def ablation(workload, legacy_cfg, new_cfg, trace=None):
    """Cumulative (legacy -> new) and leave-one-out (new with one step reverted) tables."""
    def run(cfg):
        r = evaluate(workload, cfg, trace)
        pf, dc = r["phases"]["prefill"], r["phases"]["decode"]
        traffic = {"prefill_weights_hbm_gb": pf["bytes"]["weights_hbm"] / 1e9, "prefill_act_hbm_gb": pf["bytes"]["act_hbm"] / 1e9,
                   "decode_l1_gb": dc["bytes"]["l1"] / 1e9}
        return pf["seconds"], dc["seconds"], traffic
    rows, cfg = [], copy.deepcopy(legacy_cfg)
    p0, d0, t0 = run(cfg)
    rows.append({"step": "legacy (explorer equations)", "prefill_s": p0, "decode_s": d0, **t0})
    prev = (p0, d0)
    for name, patch in ABLATION_STEPS:
        cfg = deep_merge(cfg, patch)
        p, d, t = run(cfg)
        rows.append({"step": name, "prefill_s": p, "decode_s": d, **t,
                     "prefill_delta_pct": 100 * (p / prev[0] - 1), "decode_delta_pct": 100 * (d / prev[1] - 1)})
        prev = (p, d)
    full = (prev[0], prev[1])
    loo = []
    for name, patch in ABLATION_STEPS:
        reverted = copy.deepcopy(new_cfg)
        for section, values in patch.items():
            if isinstance(values, dict):
                for k in values:
                    reverted[section][k] = legacy_cfg[section][k]
            else:
                reverted[section] = legacy_cfg[section]
        p, d, t = run(reverted)
        loo.append({"reverted": name, "prefill_s": p, "decode_s": d, **t,
                    "prefill_delta_pct": 100 * (p / full[0] - 1), "decode_delta_pct": 100 * (d / full[1] - 1)})
    return rows, loo


def switch_table(workload, cfg, trace=None):
    """Views of one design, so no modelling switch is in effect unstated: the config as given, the same hardware
    with every bracketed switch at its pessimistic setting (config.conservative), that plus the unbracketed
    alternatives as an upper bound, and the effect of moving each optimistic or informational switch on its own.

    Every evaluation keeps its `valid` flag and `errors`. An infeasible design times its unmapped ops at zero, so
    its seconds are not a result: percentage deltas are only reported when both sides of a comparison are valid."""
    def run(c):
        r = evaluate(workload, c, trace)
        return {"prefill_s": r["phases"]["prefill"]["seconds"], "decode_s": r["phases"]["decode"]["seconds"],
                "config_hash": r["config_hash"], "valid": r["valid"], "errors": r["errors"]}

    def compare(result):
        if not (given["valid"] and result["valid"]):
            return {}
        return {"prefill_delta_pct": 100 * (result["prefill_s"] / given["prefill_s"] - 1),
                "decode_delta_pct": 100 * (result["decode_s"] / given["decode_s"] - 1)}

    given = run(cfg)
    pessimistic = run(conservative(cfg))
    pessimistic.update(compare(pessimistic))
    upper = run(conservative(cfg, upper_bound=True))
    upper.update(compare(upper))
    rows = []
    for s in switch_report(cfg):
        row = {k: s[k] for k in ("path", "value", "explorer", "conservative", "alternative", "status", "optimistic",
                                 "informational", "assumes")}
        target = s["conservative"] if s["optimistic"] else s["alternative"] if s["informational"] else None
        if target is not None:
            alone = copy.deepcopy(cfg)
            set_path(alone, s["path"], target)
            r = run(alone)
            row["alone"] = r
            row.update(compare(r))
        rows.append(row)
    return {"workload": workload.name, "given": given, "conservative": pessimistic, "upper_bound": upper, "rows": rows}


# -- sensitivity ------------------------------------------------------------------------------------------------
SENSITIVITY_PARAMS = [
    "clock_mhz", "launch_cycles", "sync_cycles", "matrix.efficiency", "matrix.rates.w4", "matrix.rates.w8",
    "matrix.rates.aa", "matrix.tile_k", "vector.efficiency", "vector.special_cycles", "recurrent.efficiency",
    "dma.bytes_per_cycle", "l1.banks", "l1.bytes_per_bank_cycle", "l1.efficiency", "l1.mib", "l1.tile_fraction",
    "hbm.gbs", "hbm.efficiency", "attention.kv_tile",
]
_INT_PARAMS = {"launch_cycles", "sync_cycles", "l1.banks", "l1.bytes_per_bank_cycle", "l1.mib", "matrix.tile_k",
               "attention.kv_tile", "vector.special_cycles", "dma.bytes_per_cycle"}


def _scaled(cfg, path, factor):
    out = copy.deepcopy(cfg)
    value = get_path(out, path) * factor
    if path.endswith("efficiency"):
        value = min(1.0, value)
    if path in _INT_PARAMS:
        value = max(1, round(value))
    set_path(out, path, value)
    return out


def sensitivity(workload, cfg, params=None, delta=0.3, trace=None):
    """One-at-a-time +/- delta on each assumption. Elasticity = d ln(metric) / d ln(param)."""
    base = summarize(evaluate(workload, cfg, trace))
    rows = []
    for path in params or SENSITIVITY_PARAMS:
        x0 = get_path(cfg, path)
        if not x0:
            continue
        lo, hi = _scaled(cfg, path, 1 - delta), _scaled(cfg, path, 1 + delta)
        if validate(lo) or validate(hi):
            continue
        rlo, rhi = summarize(evaluate(workload, lo, trace)), summarize(evaluate(workload, hi, trace))
        xlo, xhi = get_path(lo, path), get_path(hi, path)
        row = {"param": path, "value": x0}
        for m in ("decode_s", "prefill_s"):
            row[f"{m}_at_low"], row[f"{m}_at_high"] = rlo[m], rhi[m]
            row[f"{m}_elasticity"] = (math.log(rhi[m] / rlo[m]) / math.log(xhi / xlo)) if xhi != xlo and rlo[m] > 0 and rhi[m] > 0 else 0.0
        row["decode_binding_flips"] = len({base["decode_binding"], rlo["decode_binding"], rhi["decode_binding"]}) > 1
        row["prefill_binding_flips"] = len({base["prefill_binding"], rlo["prefill_binding"], rhi["prefill_binding"]}) > 1
        rows.append(row)
    rows.sort(key=lambda r: -max(abs(r["decode_s_elasticity"]), abs(r["prefill_s_elasticity"])))
    return base, rows


# -- Monte-Carlo rank stability -------------------------------------------------------------------------------------
def montecarlo(workload, designs, params=None, delta=0.3, samples=100, seed=0, trace=None,
               objectives=(("cost", "min"), ("decode_s", "min"))):
    """Perturb the shared assumptions (log-uniform within +/- delta) and count how often each design stays on the
    (cost, decode time) frontier. designs: {name: config}. Costs are held fixed so this measures performance uncertainty."""
    rng = random.Random(seed)
    params = params or SENSITIVITY_PARAMS
    tally = {n: {"frontier": 0, "decode": [], "prefill": []} for n in designs}
    for _ in range(samples):
        factors = {p: math.exp(rng.uniform(math.log(1 - delta), math.log(1 + delta))) for p in params}
        recs = {}
        for name, cfg in designs.items():
            c = copy.deepcopy(cfg)
            for p, f in factors.items():
                if get_path(c, p):
                    c = _scaled(c, p, f)
            if validate(c):
                continue
            s = summarize(evaluate(workload, c, trace))
            s["name"] = name
            recs[name] = s
            tally[name]["decode"].append(s["decode_s"])
            tally[name]["prefill"].append(s["prefill_s"])
        for r in pareto(list(recs.values()), objectives):
            tally[r["name"]]["frontier"] += 1
    out = {}
    for name, t in tally.items():
        def q(v, f):
            v = sorted(v)
            return v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")
        out[name] = {"frontier_frequency": t["frontier"] / samples,
                     "decode_p05_s": q(t["decode"], 0.05), "decode_p50_s": q(t["decode"], 0.5), "decode_p95_s": q(t["decode"], 0.95),
                     "prefill_p05_s": q(t["prefill"], 0.05), "prefill_p50_s": q(t["prefill"], 0.5), "prefill_p95_s": q(t["prefill"], 0.95)}
    return out
