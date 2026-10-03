"""Offload/compute-unit boundary comparison: named component groups vs. structural boundary bytes.

Reuses ops.py's software-facing grouping (MLP projections, Vocabulary output head, Attention/
DeltaNet projections, DeltaNet recurrence, Attention QK/softmax/AV, Convolution, Copies/gather/
state movement, Vector/norm/activation/other) so this lines up with the measured CPU time shares
already reported in operation-summary.json. For each group, in one captured graph
(prefill.json/decode.json), this computes:

- matrix MACs (measured from the graph, MUL_MAT/SSM_CONV only, same convention as graph.py)
- boundary bytes: logical bytes of every tensor that crosses INTO or OUT OF the group, i.e. the
  data a candidate offload of that group would have to pack/transfer/synchronize regardless of
  the eventual hardware -- a structural lower bound, not a measured transfer time
- leaf-input bytes: logical bytes of every distinct unscheduled tensor the group reads (weights,
  KV/recurrent cache, graph inputs). These are NOT in boundary bytes, which only count edges
  between scheduled ops; a weight-heavy group (e.g. the vocabulary head) moves far more than its
  boundary bytes suggest
- arithmetic ops (2x MACs) per boundary byte and per (boundary + leaf-input) byte: structural
  ratios of logical sizes. Neither is a measured or modelled memory traffic figure; reuse,
  tiling, caching and quantized weight formats decide the real traffic

and, when a matching per-run/per-prompt CPU operation trace is given, the measured share of CPU
time that group actually took -- real numbers, not an accelerator estimate.

Everything here is either read from the captured graph or the trace. The only assumption this
tool ever introduces is an OPTIONAL illustrative transfer time from --bandwidth-gbps/--dispatch-us,
which is left out entirely unless the caller supplies it, and is always reported next to the
measured/structural figures, never in place of them.
"""
import argparse
import collections
import json
from pathlib import Path

from fusion import PHASE_RE
from graph import macs_of
from ops import group as group_of_trace
from result_io import exists, open_result, output_dir, write_json, write_text


def group_of_node(t, tensors):
    """Same taxonomy as ops.py's group(), adapted to a captured-graph tensor record."""
    op = t["op"]
    by_slot = {s["slot"]: s["tensor"] for s in t["sources"]}
    w = tensors[by_slot[0]]["name"] if 0 in by_slot and by_slot[0] in tensors else ""
    if op == "MUL_MAT":
        if "ffn_" in w:
            return "MLP projections"
        if w in ("output.weight", "token_embd.weight"):
            return "Vocabulary output head"
        if ".weight" in w or w.endswith(".weight (repacked)"):
            return "Attention/DeltaNet projections"
        return "Other matrix products"
    if op == "GATED_DELTA_NET":
        return "DeltaNet recurrence"
    if op == "FLASH_ATTN_EXT":
        return "Attention QK/softmax/AV"
    if op == "SSM_CONV":
        return "Convolution"
    if op in ("GET_ROWS", "SET_ROWS", "CPY", "CONT", "CONCAT", "DUP"):
        return "Copies/gather/state movement"
    return "Vector/norm/activation/other"


def structural_boundaries(tensors, order):
    """Per group: node count, matrix MACs, and bytes crossing into/out of the group."""
    pos = {i: idx for idx, i in enumerate(order)}
    groups = {i: group_of_node(tensors[i], tensors) for i in order}
    nodes, macs = collections.Counter(), collections.Counter()
    in_bytes, out_bytes = collections.Counter(), collections.Counter()
    in_edges, out_edges = collections.Counter(), collections.Counter()
    leaf_bytes, leaf_seen = collections.Counter(), set()
    for i in order:
        t, g = tensors[i], groups[i]
        nodes[g] += 1
        m = macs_of(t, tensors)
        if m is not None:
            macs[g] += m
        for s in t["sources"]:
            p = s["tensor"]
            if p not in pos:
                if (g, p) not in leaf_seen and p in tensors:
                    leaf_seen.add((g, p))
                    leaf_bytes[g] += tensors[p]["logical_bytes"]
                continue
            if groups[p] != g:
                in_bytes[g] += tensors[p]["logical_bytes"]
                in_edges[g] += 1
                out_bytes[groups[p]] += tensors[p]["logical_bytes"]
                out_edges[groups[p]] += 1
    return {g: {"nodes": nodes[g], "matrix_macs": macs[g],
                "boundary_in_bytes": in_bytes[g], "boundary_in_edges": in_edges[g],
                "boundary_out_bytes": out_bytes[g], "boundary_out_edges": out_edges[g],
                "leaf_input_bytes": leaf_bytes[g]}
            for g in nodes}


def measured_time_by_group(op_trace_path, phase, prompt_tokens=None):
    """Sum elapsed_us by group across an entire per-prompt or fixed-length CPU op trace.
    phase: 'prefill' or 'decode'. Returns (totals, total_us) or (None, 0) if no trace given.
    prompt_tokens: required when op_trace_path may cover more than one prompt length (the
    fixed-length trace mixes 128/512/2048 in one file); a per-prompt trace has only one length."""
    if op_trace_path is None or not exists(op_trace_path):
        return None, 0
    want_prefix = "prefill" if phase == "prefill" else "decode"
    totals, total_us = collections.Counter(), 0
    with open_result(op_trace_path) as stream:
        for line in stream:
            x = json.loads(line)
            m = PHASE_RE.fullmatch(x["phase"])
            if not m or not m[3].startswith(want_prefix) or x["graph_status"] != 0 or x["elapsed_us"] < 0:
                continue
            if prompt_tokens is not None and int(m[1]) != prompt_tokens:
                continue
            totals[group_of_trace(x)] += x["elapsed_us"]
            total_us += x["elapsed_us"]
    return totals, total_us


def analyze_phase(graph_path, op_trace_path, phase, bandwidth_gbps, dispatch_us):
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    tensors = {t["id"]: t for t in graph["tensors"]}
    order = graph["scheduled_nodes"]
    boundaries = structural_boundaries(tensors, order)
    measured, total_us = measured_time_by_group(op_trace_path, phase, graph["metadata"]["prompt_tokens"])
    rows = []
    for g, b in boundaries.items():
        crossing_bytes = b["boundary_in_bytes"] + b["boundary_out_bytes"]
        row = {
            "group": g, "nodes": b["nodes"], "matrix_macs": b["matrix_macs"],
            "arithmetic_ops": 2 * b["matrix_macs"],
            "boundary_in_bytes": b["boundary_in_bytes"], "boundary_out_bytes": b["boundary_out_bytes"],
            "boundary_crossing_edges": b["boundary_in_edges"] + b["boundary_out_edges"],
            "arithmetic_ops_per_boundary_byte": (2 * b["matrix_macs"] / crossing_bytes) if crossing_bytes else None,
            "leaf_input_bytes": b["leaf_input_bytes"],
            "arithmetic_ops_per_boundary_and_leaf_byte": ((2 * b["matrix_macs"] / (crossing_bytes + b["leaf_input_bytes"]))
                                                          if crossing_bytes + b["leaf_input_bytes"] else None),
        }
        if measured is not None:
            us = measured.get(g, 0)
            row["measured_cpu_us"] = us
            row["measured_cpu_time_percent"] = (us / total_us * 100) if total_us else None
        if bandwidth_gbps:
            row["illustrative_transfer_us_at_given_bandwidth"] = crossing_bytes / (bandwidth_gbps * 1e9) * 1e6
        if dispatch_us:
            row["illustrative_dispatch_us_at_given_latency"] = row["boundary_crossing_edges"] * dispatch_us
        rows.append(row)
    rows.sort(key=lambda r: -(r.get("measured_cpu_us") or r["matrix_macs"]))
    return rows


def _report(out_dir, phases, has_trace, has_assumptions):
    lines = ["# Offload/compute-unit boundary comparison", "",
             "Groups follow ops.py's software-facing taxonomy so they line up with "
             "operation-summary.json. Boundary bytes are the logical size of every tensor "
             "crossing into or out of a group in the scheduled dataflow -- what a candidate "
             "offload of that group would have to pack/transfer/synchronize, not a measured "
             "transfer time. Leaf-input bytes are the distinct weights, caches and graph inputs the "
             "group reads; they are not part of boundary bytes. Both ops/byte columns are ratios of "
             "logical sizes from the graph (structural), not measured or modelled memory traffic.", ""]
    if not has_trace:
        lines.append("No matching CPU operation trace was given; only structural figures are shown.")
    if has_assumptions:
        lines.append("Columns starting `illustrative_` use a caller-supplied bandwidth/dispatch "
                      "figure for orientation only; they are not measured and are not an ASIC estimate.")
    lines.append("")
    for phase, rows in phases.items():
        lines += [f"## {phase}", "",
                  "| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Leaf-input bytes | "
                  "Ops/boundary byte | Ops/(boundary+leaf) byte | " + ("Measured CPU time" if has_trace else "") + " |",
                  "|---|---:|---:|---:|---:|---:|---:|" + (":--|" if has_trace else "")]
        for r in rows:
            aib = f"{r['arithmetic_ops_per_boundary_byte']:.2f}" if r["arithmetic_ops_per_boundary_byte"] else "n/a"
            ail = f"{r['arithmetic_ops_per_boundary_and_leaf_byte']:.2f}" if r["arithmetic_ops_per_boundary_and_leaf_byte"] else "n/a"
            measured = f"{r['measured_cpu_time_percent']:.1f}%" if has_trace and r.get("measured_cpu_time_percent") is not None else ""
            lines.append(f"| {r['group']} | {r['nodes']} | {r['matrix_macs']:,} | "
                         f"{r['boundary_in_bytes'] + r['boundary_out_bytes']:,} | {r['leaf_input_bytes']:,} | {aib} | {ail} |" +
                         (f" {measured} |" if has_trace else ""))
        lines.append("")
    lines += ["## Limitations", "",
              "- Boundary bytes come from one captured graph per phase; different prompt lengths "
              "keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, "
              "so absolute bytes scale with prompt length even though the ranking usually will not.",
              "- Measured CPU time is a CPU interval share on this host, not accelerator latency, "
              "area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).",
              "- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more "
              "or less resident than this structural count implies.",
              "- Leaf-input bytes use the captured (mixed Q4_K_M) storage size of each weight and count each "
              "tensor once per group per graph; they say nothing about on-chip buffering, SRAM traffic or stalls.", ""]
    write_text(out_dir / "OFFLOAD-REPORT.md", "\n".join(lines))


def analyze_directory(graph_dir, op_trace_path=None, out_dir=None, bandwidth_gbps=None, dispatch_us=None):
    graph_dir = Path(graph_dir).resolve()
    out_dir = output_dir(Path(out_dir)) if out_dir else output_dir(graph_dir)
    phases = {}
    for phase in ("prefill", "decode"):
        rows = analyze_phase(graph_dir / f"{phase}.json", op_trace_path, phase, bandwidth_gbps, dispatch_us)
        write_json(out_dir / f"{phase}.offload-boundaries.json", rows, trailing_newline=True)
        phases[phase] = rows
    _report(out_dir, phases, op_trace_path is not None, bool(bandwidth_gbps or dispatch_us))
    return phases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("graph_dir", type=Path, help="Directory holding prefill.json and decode.json")
    parser.add_argument("--op-trace", type=Path,
                        help="cpu-op-trace.jsonl(.gz) from the run that captured this graph, "
                             "for measured per-group CPU time share")
    parser.add_argument("--output-dir", type=Path, help="Default: the graph directory itself")
    parser.add_argument("--bandwidth-gbps", type=float,
                        help="Illustrative only: an assumed transfer bandwidth, not a measured one")
    parser.add_argument("--dispatch-us", type=float,
                        help="Illustrative only: an assumed per-crossing dispatch latency")
    args = parser.parse_args(argv)
    phases = analyze_directory(args.graph_dir, args.op_trace, args.output_dir,
                               args.bandwidth_gbps, args.dispatch_us)
    print(json.dumps({p: len(r) for p, r in phases.items()}, indent=2))


if __name__ == "__main__":
    main()
