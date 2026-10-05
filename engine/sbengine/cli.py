"""Command line: python -m sbengine <command> [options]   (run from the engine/ directory)

  list                      workloads in a run
  eval                      one workload, one profile: timing, bytes, bounds, binding resource
  sweep                     grid over config paths -> CSV + meta.json (+ Pareto front)
  ablation                  switch each equation from the explorer's version to this engine's
  switches                  every modelling switch: its setting, what it assumes, and the result without it
  groups                    parallel projection groups and anchored fusion groups of one captured workload
  sensitivity               +/- delta on each assumption; which ones decide the answer
  montecarlo                rank stability of several designs under assumption uncertainty
  parity                    MAC/attention accounting checks against the captured summaries
  validate-cpu              held-out check of the work/traffic accounting against measured CPU op timings
  serve                     local web page: move the sliders, re-run the real model  (http://127.0.0.1:8765)
"""
import argparse
import json
import sys
from pathlib import Path

from . import VERSION
from .config import apply_sets, load_profile
from .graph import has_capture, latest_run, load_workload
from .model import evaluate
from .trace import load_trace

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _value(text):
    """A sweep value: JSON if it parses (numbers, true/false, null), else the bare string (e.g. via_l1)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text.strip()


def _ms(s):
    return f"{s * 1e3:10.3f} ms"


def _workload(args):
    run = Path(args.run) if args.run else latest_run(RESULTS)
    return run, load_workload(run / "graphs" / args.graph)


def _config(args):
    cfg, doc = load_profile(args.profile)
    return apply_sets(cfg, getattr(args, "set", None)), doc


def _add_common(p, graph="pp512-fa-on"):
    p.add_argument("--run", help="results/<run> (default: newest run with graphs/)")
    p.add_argument("--graph", default=graph, help=f"graph folder name (default {graph})")
    p.add_argument("--profile", default="accel-balanced", help="profiles/<name>.json or a path")
    p.add_argument("--set", action="append", metavar="PATH=VALUE", help="override a config value, e.g. matrix.count=2")


def cmd_list(args):
    run = Path(args.run) if args.run else latest_run(RESULTS)
    print(f"run: {run.name}")
    for d in sorted((run / "graphs").glob("*")):
        if has_capture(d):
            print(" ", d.name)


def cmd_eval(args):
    run, w = _workload(args)
    cfg, doc = _config(args)
    trace = load_trace(run, w) if cfg["host"]["enabled"] else None
    r = evaluate(w, cfg, trace, detail=bool(args.nodes))
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(f"engine {VERSION}  workload {w.name} ({r['prompt_tokens']} tokens, flash attention {r['flash_attention']})  "
          f"profile {doc.get('name')}  config {r['config_hash']}  batch {r['batch']}")
    res = r["resources"]
    print(f"resources: cost {res['cost']:.0f}  DSP {res['dsp']:.0f} / {res['dsp_limit']:.0f}  valid={r['valid']}")
    for e in r["errors"]:
        print("  !", e)
    optimistic = [s for s in r["switches"] if s["optimistic"]]
    if optimistic:
        print("assumes (switches off their pessimistic setting): "
              + ", ".join(f"{s['path']}={json.dumps(s['value'])}" for s in optimistic))
        print("  `python -m sbengine switches` shows what each one is worth")
    for phase in ("prefill", "decode"):
        p = r["phases"][phase]
        b, by = p["bytes"], p["time_by_bound"]
        bound = max(by, key=by.get) if by else "-"
        print(f"\n{phase}: {_ms(p['seconds'])}   (lower bound {_ms(p['bounds']['lower_bound'])}, serial {_ms(p['bounds']['serial'])})"
              f"   {p['tokens_per_s']:.1f} tok/s   mostly {bound}-bound")
        print(f"  HBM  weights {b['weights_hbm'] / 1e9:8.3f} GB  KV/state {b['kv_state_hbm'] / 1e9:7.3f} GB  activations {b['act_hbm'] / 1e9:7.3f} GB"
              f"   | L1 {b['l1'] / 1e9:8.3f} GB")
        print(f"  service: L1 {_ms(p['l1_service_s'])}  HBM {_ms(p['hbm_service_s'])}  pools "
              + "  ".join(f"{k} {v * 1e3:.2f} ms" for k, v in sorted(p["pool_busy"].items())))
        f = p["footprint"]
        print(f"  HBM footprint {f['hbm_required'] / 2**30:.3f} GiB (weights {f['weights'] / 2**30:.3f}, KV {f['kv'] / 2**30:.3f}, "
              f"state {f['state'] / 2**30:.3f}, spilled activations {f['peak_spilled_activations'] / 2**30:.3f})")
        top = sorted(p["by_op"].items(), key=lambda kv: -kv[1]["serial_s"])[:6]
        print("  top ops: " + ", ".join(f"{k} {v['serial_s'] * 1e3:.2f} ms" for k, v in top))
        if args.nodes:
            print(f"  slowest {args.nodes} nodes:")
            for row in sorted(p["rows"], key=lambda x: -x["duration_s"])[:args.nodes]:
                print(f"    {row['name'][:32]:32s} {row['op']:16s} {row['pool'] or '-':9s} {row['duration_s'] * 1e6:10.1f} us  ({row['dominant']})  {row['note']}")


def cmd_sweep(args):
    from . import sweep
    run, w = _workload(args)
    base, doc = _config(args)
    axes = {}
    for item in args.axis:
        key, _, raw = item.partition("=")
        axes[key.strip()] = [_value(v) for v in raw.split(",")]
    trace = load_trace(run, w) if base["host"]["enabled"] else None
    records = sweep.run_sweep(w, base, axes, trace)
    path = sweep.write_sweep(args.out, args.name, records, base, axes, w, " ".join(sys.argv))
    front = sweep.pareto(records, (("cost", "min"), ("decode_s", "min")))
    print(f"{len(records)} points -> {path}  ({sum(1 for r in records if r.get('valid'))} valid, {len(front)} on the cost/decode Pareto front)")
    for r in sorted(front, key=lambda r: r["cost"]):
        print("  cost %8.0f  decode %9.3f ms  prefill %9.1f ms  %s  (%s-bound)" % (
            r["cost"], r["decode_s"] * 1e3, r["prefill_s"] * 1e3, {k: r[k] for k in axes}, r["decode_binding"]))


def cmd_ablation(args):
    from . import sweep
    run, w = _workload(args)
    legacy, _ = load_profile("legacy-equations")
    new, _ = load_profile(args.profile)
    rows, loo = sweep.ablation(w, legacy, new)
    if args.json:
        Path(args.json).write_text(json.dumps({"workload": w.name, "cumulative": rows, "leave_one_out": loo}, indent=2) + "\n", encoding="utf-8")
    print(f"ablation on {w.name} (profile {args.profile})\n\nCUMULATIVE: explorer equations -> this engine")
    for r in rows:
        d = "" if "prefill_delta_pct" not in r else f"  ({r['prefill_delta_pct']:+6.1f}% / {r['decode_delta_pct']:+6.1f}%)"
        print(f"  {r['step'][:74]:74s} prefill {r['prefill_s'] * 1e3:9.1f} ms  decode {r['decode_s'] * 1e3:8.2f} ms{d}"
              f"   | prefill weights HBM {r['prefill_weights_hbm_gb']:6.2f} GB, act HBM {r['prefill_act_hbm_gb']:6.2f} GB; decode L1 {r['decode_l1_gb']:5.2f} GB")
    print("\nLEAVE-ONE-OUT: revert one equation from the full engine")
    for r in loo:
        print(f"  {r['reverted'][:74]:74s} prefill {r['prefill_delta_pct']:+7.1f}%  decode {r['decode_delta_pct']:+7.1f}%")


def cmd_switches(args):
    from . import sweep
    run, w = _workload(args)
    cfg, doc = _config(args)
    trace = load_trace(run, w) if cfg["host"]["enabled"] else None
    t = sweep.switch_table(w, cfg, trace)
    if args.json:
        Path(args.json).write_text(json.dumps(t, indent=2) + "\n", encoding="utf-8")
    g, c = t["given"], t["conservative"]

    def show(v):
        return "-" if v is None else json.dumps(v)

    print(f"modelling switches on {w.name} (profile {doc.get('name')})")
    print()
    print(f"  {'as configured':36s} prefill {_ms(g['prefill_s'])}  decode {_ms(g['decode_s'])}   config {g['config_hash']}")
    print(f"  {'every bracketed switch pessimistic':36s} prefill {_ms(c['prefill_s'])}  decode {_ms(c['decode_s'])}   config {c['config_hash']}"
          f"   ({100 * (c['prefill_s'] / g['prefill_s'] - 1):+.1f}% / {100 * (c['decode_s'] / g['decode_s'] - 1):+.1f}%)")
    print()
    print(f"  {'switch':28s} {'value':>14s} {'pessimistic':>12s} {'explorer':>14s}  {'status':20s} alone: prefill / decode")
    for r in t["rows"]:
        delta = f"{r['prefill_delta_pct']:+7.1f}% / {r['decode_delta_pct']:+6.1f}%" if "prefill_delta_pct" in r else ""
        mark = "*" if r["optimistic"] else " "
        print(f" {mark}{r['path']:28s} {show(r['value']):>14s} {show(r['conservative']):>12s} {show(r['explorer']):>14s}  {r['status']:20s} {delta}")
    print()
    print("  * off its pessimistic setting. 'alone' is the change from moving just that switch back. A switch with no")
    print("    pessimistic setting is a derived accounting change or a design choice (docs/EQUATIONS.md).")
    print("  Estimates under assumed hardware parameters; a switch makes an assumption visible, it does not validate it.")


def cmd_groups(args):
    from . import groups
    run, w = _workload(args)
    out = {phase: groups.summary(w.graph(phase)) for phase in ("prefill", "decode")}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"anchored fusion groups on {w.name} (rule: sbengine/groups.py; candidates, not implemented kernels)")
    for phase, s in out.items():
        print()
        print(f"{phase}: {s['nodes']} ops, {s['parallel_groups']} parallel groups, {s['fusion_groups']} fusion groups, "
              f"{s['fused_edges']} intermediates kept inside a group")
        for kind, t in s["layer_types"].items():
            print(f"  {kind:15s} x{t['layers']:<3d} per layer: {t['ops_per_layer']:3d} ops, {t['parallel_groups_per_layer']} parallel groups, "
                  f"{t['fusion_groups_per_layer']:2d} fusion groups")


def cmd_sensitivity(args):
    from . import sweep
    run, w = _workload(args)
    cfg, _ = _config(args)
    base, rows = sweep.sensitivity(w, cfg, delta=args.delta)
    if args.json:
        Path(args.json).write_text(json.dumps({"base": base, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"sensitivity on {w.name}: base prefill {base['prefill_s'] * 1e3:.1f} ms ({base['prefill_binding']}-bound), "
          f"decode {base['decode_s'] * 1e3:.2f} ms ({base['decode_binding']}-bound); +/-{args.delta:.0%} per parameter")
    print(f"  {'parameter':28s} {'value':>9s} {'decode elasticity':>18s} {'prefill elasticity':>19s}  binding flips")
    for r in rows:
        flips = ("decode " if r["decode_binding_flips"] else "") + ("prefill" if r["prefill_binding_flips"] else "")
        print(f"  {r['param']:28s} {r['value']:9.4g} {r['decode_s_elasticity']:18.2f} {r['prefill_s_elasticity']:19.2f}  {flips}")


def cmd_montecarlo(args):
    from . import sweep
    run, w = _workload(args)
    designs = {name: load_profile(name)[0] for name in args.designs}
    out = sweep.montecarlo(w, designs, delta=args.delta, samples=args.samples, seed=args.seed)
    print(f"Monte-Carlo on {w.name}: {args.samples} samples, assumptions perturbed +/-{args.delta:.0%} (log-uniform), seed {args.seed}")
    for name, r in out.items():
        print(f"  {name:20s} on cost/decode frontier {r['frontier_frequency']:6.1%}   decode p05/p50/p95 "
              f"{r['decode_p05_s'] * 1e3:7.2f}/{r['decode_p50_s'] * 1e3:7.2f}/{r['decode_p95_s'] * 1e3:7.2f} ms   prefill p50 {r['prefill_p50_s'] * 1e3:8.1f} ms")


def cmd_parity(args):
    from .graph import Graph
    run = Path(args.run) if args.run else latest_run(RESULTS)
    cfg, _ = load_profile("accel-balanced")
    bad = 0
    for d in sorted((run / "graphs").glob("*")):
        if not has_capture(d):
            continue
        w = load_workload(d)
        r = evaluate(w, cfg)
        for phase in ("prefill", "decode"):
            summary = json.loads((d / f"{phase}.summary.json").read_text(encoding="utf-8"))["macs_by_supported_op"] if (d / f"{phase}.summary.json").is_file() else None
            mine = r["phases"][phase]["macs_by_op"]
            if summary is None:
                continue
            ok = all(abs(mine.get(op, 0) - v) < 0.5 for op, v in summary.items())
            bad += not ok
            print(f"  {'ok ' if ok else 'BAD'} {d.name:28s} {phase:8s} MUL_MAT {mine.get('MUL_MAT', 0):.0f} vs {summary.get('MUL_MAT')}")
    print("parity:", "all match" if not bad else f"{bad} mismatches")
    return 1 if bad else 0


def cmd_validate_cpu(args):
    from . import cpu_validate
    run = Path(args.run) if args.run else latest_run(RESULTS)
    res = cpu_validate.run(run)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"train on {res['train']}; predict {len(res['test'])} held-out per-prompt graphs (only phases with a comparable trace)")
    for label, rows in (("training fit", res["train_fit"]), ("HELD-OUT", res["heldout"])):
        print(f"\n{label}")
        for r in rows:
            print(f"  {r['workload']:28s} {r['phase']:8s} {r['tokens']:4d} tok  measured {r['measured_s'] * 1e3:9.1f} ms  predicted {r['predicted_s'] * 1e3:9.1f} ms  {r['error_pct']:+6.1f}%")
    for phase in ("prefill", "decode"):
        e = [abs(r["error_pct"]) for r in res["heldout"] if r["phase"] == phase]
        if e:
            print(f"held-out {phase}: mean |error| {sum(e) / len(e):.1f}%  max {max(e):.1f}%  (n={len(e)})")
    print("\nThis validates the MAC/byte accounting, not the accelerator timing equations.")


def cmd_serve(args):
    from .web import serve
    serve(args.run, args.host, args.port, args.open, RESULTS)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sbengine", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--run"); p.set_defaults(fn=cmd_list)
    p = sub.add_parser("eval"); _add_common(p); p.add_argument("--json"); p.add_argument("--nodes", type=int, default=0); p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("sweep"); _add_common(p)
    p.add_argument("--axis", action="append", required=True, metavar="PATH=v1,v2,...")
    p.add_argument("--out", default="out"); p.add_argument("--name", default="sweep"); p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("ablation"); _add_common(p, "pp512-fa-off"); p.add_argument("--json"); p.set_defaults(fn=cmd_ablation)
    p = sub.add_parser("switches"); _add_common(p); p.add_argument("--json"); p.set_defaults(fn=cmd_switches)
    p = sub.add_parser("groups"); _add_common(p); p.add_argument("--json"); p.set_defaults(fn=cmd_groups)
    p = sub.add_parser("sensitivity"); _add_common(p); p.add_argument("--delta", type=float, default=0.3); p.add_argument("--json"); p.set_defaults(fn=cmd_sensitivity)
    p = sub.add_parser("montecarlo"); _add_common(p)
    p.add_argument("--designs", nargs="+", default=["accel-balanced", "accel-efficient"])
    p.add_argument("--delta", type=float, default=0.3); p.add_argument("--samples", type=int, default=100); p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_montecarlo)
    p = sub.add_parser("parity"); p.add_argument("--run"); p.set_defaults(fn=cmd_parity)
    p = sub.add_parser("validate-cpu"); p.add_argument("--run"); p.add_argument("--json"); p.set_defaults(fn=cmd_validate_cpu)
    p = sub.add_parser("serve"); p.add_argument("--run"); p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765); p.add_argument("--open", action="store_true", help="open the page in a browser")
    p.set_defaults(fn=cmd_serve)
    args = parser.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
