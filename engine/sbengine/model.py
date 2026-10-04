"""evaluate(): one workload (captured prefill + decode graphs) under one hardware/assumption config.

Pipeline per phase:
  1. per-node compute time and traffic (costs.py)
  2. fusion: drop single-consumer epilogue intermediates (memory.fused_roots)
  3. activation residency: which intermediates stay in L1, what round-trips through HBM (memory.py)
  4. node duration = max(compute, L1 service, HBM service) + launch   (or their sum if within_op_overlap is off)
  5. whole-graph time: serial sum, dependency-aware list schedule, lower bound (schedule.py)
Feasibility: every op mapped, matmul tile fits, DSP budget, HBM capacity (weights + KV + state + spilled activations).
"""
import math

from . import VERSION
from .config import config_hash
from .costs import Ctx, node_cost
from .memory import drop_fused, fused_roots, per_op_spill, simulate_residency
from .resources import resources
from .schedule import schedule

GiB = 2**30


def _footprint(ctx, g, cfg):
    weights = sum(ctx.weight_bytes(g, g.tensors[r]) for r in g.roots_of_kind("weight"))
    ratio = 1.0
    cap = cfg["attention"]["capacity_tokens"]
    if cap:
        ratio = cap / max(1, g.meta["context_capacity"])
    kv = sum(g.tensors[r].elements for r in g.roots_of_kind("kv")) * ctx.kv_b * ctx.B * ratio
    state = sum(g.tensors[r].elements for r in g.roots_of_kind("state")) * ctx.state_b * ctx.B
    return weights, kv, state


def _host_fallback(ctx, cfg, g, node, cost, trace_phase):
    """Replace a 'no implementation' node by its measured CPU time plus a host link transfer."""
    host = cfg["host"]
    if not host["enabled"] or cost.pool is not None or trace_phase is None:
        return
    seconds = trace_phase.get(node.spos)
    if seconds is None:
        return
    moved = sum(b for _, b, _ in cost.reads) + sum(b for _, b in cost.writes) + cost.w_hbm + cost.p_hbm_r
    cost.pool = "Host"
    cost.compute_s = seconds + moved / (host["link_gbps"] * 1e9) + 2 * host["link_latency_us"] * 1e-6
    cost.error = None
    cost.note = "host fallback (measured CPU time + link)"


def evaluate_phase(ctx, g, cfg, trace_phase=None, detail=False):
    weights, kv, state = _footprint(ctx, g, cfg)
    state_seq = sum(g.tensors[r].elements for r in g.roots_of_kind("state")) * ctx.state_b
    ctx.state_resident = min(1.0, ctx.state_cap / (state_seq * ctx.B)) if state_seq else 1.0

    costs = [node_cost(ctx, g, n) for n in g.nodes]
    for n, c in zip(g.nodes, costs):
        _host_fallback(ctx, cfg, g, n, c, trace_phase)
    errors = sorted({c.error for c in costs if c.error})
    for c in costs:
        if c.error:
            c.compute_s = 0.0

    legacy = cfg["memory"]["residency"] == "per_op"
    if not legacy:
        drop_fused(costs, fused_roots(g, cfg["memory"]["fusion"]))
        act_r, act_w, peak_spilled, stats = simulate_residency(costs, ctx.act_cap)
        act_bytes = [r + w for r, w in zip(act_r, act_w)]
    else:
        spill, peak_spilled = per_op_spill(costs, ctx.l1_bytes * (1 - cfg["l1"]["state_reserve"]))
        act_bytes, stats = spill, {"hits": 0, "misses": 0}

    overlap = cfg["schedule"]["within_op_overlap"]
    durations, pools, l1_s, hbm_s, aux, rows = [], [], [], [], [], []
    tot = {"weights_hbm": 0.0, "kv_state_hbm": 0.0, "input_hbm": 0.0, "act_hbm": 0.0, "l1": 0.0, "macs": 0.0, "arith": 0.0}
    macs_by_op, by_op = {}, {}
    bound_count = {}
    for n, c, act in zip(g.nodes, costs, act_bytes):
        hbm = c.w_hbm + c.p_hbm_r + c.p_hbm_w + c.in_hbm + act
        l1 = sum(c.feed.values()) + sum(c.out.values()) + c.l1_fixed + hbm - c.bypass
        ts_l1, ts_hbm = l1 / ctx.l1_bw, hbm / ctx.hbm_bw
        parts = (c.compute_s, ts_l1, ts_hbm)
        d = (max(parts) if overlap else sum(parts)) + ctx.launch_s
        durations.append(d)
        pools.append(c.pool or "None")
        l1_s.append(ts_l1)
        hbm_s.append(ts_hbm)
        aux.append(c.aux_busy)
        tot["weights_hbm"] += c.w_hbm
        tot["kv_state_hbm"] += c.p_hbm_r + c.p_hbm_w
        tot["input_hbm"] += c.in_hbm
        tot["act_hbm"] += act
        tot["l1"] += l1
        tot["macs"] += c.macs
        tot["arith"] += c.arith
        macs_by_op[n.op] = macs_by_op.get(n.op, 0.0) + c.macs
        dominant = "hbm" if ts_hbm >= max(c.compute_s, ts_l1) else "l1" if ts_l1 >= c.compute_s else (c.pool or "none").lower()
        bound_count[dominant] = bound_count.get(dominant, 0) + d
        agg = by_op.setdefault(n.op, {"count": 0, "serial_s": 0.0, "hbm_bytes": 0.0, "l1_bytes": 0.0, "macs": 0.0})
        agg["count"] += 1
        agg["serial_s"] += d
        agg["hbm_bytes"] += hbm
        agg["l1_bytes"] += l1
        agg["macs"] += c.macs
        if detail:
            rows.append({"node": n.idx, "spos": n.spos, "name": n.tensor.name, "op": n.op, "pool": c.pool,
                         "compute_s": c.compute_s, "l1_s": ts_l1, "hbm_s": ts_hbm, "duration_s": d,
                         "hbm_bytes": hbm, "l1_bytes": l1, "macs": c.macs, "dominant": dominant, "note": c.note})
    sched = schedule(durations, pools, g_deps(g), l1_s, hbm_s, ctx.sync_s, aux, detail=detail)
    seconds = sched["list"] if cfg["schedule"]["mode"] == "pools" else sched["serial"]
    tokens = g.meta["input_tokens"] * ctx.B
    out = {
        "seconds": seconds, "bounds": {k: sched[k] for k in ("serial", "list", "lower_bound", "critical_path")},
        "pool_busy": sched["pool_busy"], "l1_service_s": sched["l1_total"], "hbm_service_s": sched["hbm_total"],
        "tokens_per_s": tokens / seconds if seconds else math.inf,
        "per_sequence_tps": 1.0 / seconds if seconds else math.inf,
        "bytes": tot, "macs_by_op": macs_by_op, "by_op": by_op,
        "time_by_bound": bound_count,
        "footprint": {"weights": weights, "kv": kv, "state": state, "peak_spilled_activations": peak_spilled,
                      "hbm_required": weights + kv + state + peak_spilled},
        "state_resident_fraction": ctx.state_resident, "residency": stats,
        "fused_edges": len(fused_roots(g, cfg["memory"]["fusion"])) if not legacy else 0,
        "nodes": len(g.nodes), "errors": errors,
    }
    if detail:
        out["rows"] = rows
        out["pool_schedule"] = {"operations": sched["operations"], "reservations": sched["reservations"]}
    return out


def g_deps(g):
    if not hasattr(g, "_deps"):
        g._deps = g.deps()
    return g._deps


def evaluate(workload, cfg, trace=None, detail=False):
    """Evaluate prefill and decode. `trace` (from trace.load_trace) is only needed for host fallback."""
    ctx = Ctx(cfg)
    res = resources(cfg)
    phases, errors = {}, []
    for phase in ("prefill", "decode"):
        phases[phase] = evaluate_phase(ctx, workload.graph(phase), cfg, (trace or {}).get(phase), detail)
        errors += phases[phase]["errors"]
    if res["dsp"] > res["dsp_limit"]:
        errors.append("Estimated DSP demand exceeds the configured usable budget.")
    need = max(p["footprint"]["hbm_required"] for p in phases.values())
    if need > cfg["hbm"]["gib"] * GiB:
        errors.append(f"HBM footprint {need / GiB:.2f} GiB exceeds capacity {cfg['hbm']['gib']} GiB.")
    return {"engine_version": VERSION, "config_hash": config_hash(cfg), "workload": workload.name,
            "prompt_tokens": workload.meta["prompt_tokens"], "flash_attention": workload.meta.get("flash_attention"),
            "batch": cfg["batch"], "valid": not errors, "errors": sorted(set(errors)), "resources": res,
            "phases": phases}
