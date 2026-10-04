"""Aggregate CPU operation traces; timing for excluded cases stays diagnostic."""
import collections
import csv
import json
import math
import re

from policy import timing_eligible
from result_io import exists, open_csv, open_result, parse_paths, write_json


def _lines(path):
    """Stream a (possibly gzipped, possibly very large) trace and close the file when done."""
    with open_result(path) as stream:
        yield from stream


def group(x):
    t = x["tensor"]
    o = t["op"]
    src = x["src"]
    w = src[0]["name"] if src and src[0] else ""
    if o == "MUL_MAT":
        if "ffn_" in w:
            return "MLP projections"
        if w in ["output.weight", "token_embd.weight"]:
            return "Vocabulary output head"
        if ".weight" in w or w.endswith(".weight (repacked)"):
            return "Attention/DeltaNet projections"
        return "Other matrix products"
    if o == "GATED_DELTA_NET":
        return "DeltaNet recurrence"
    if o == "FLASH_ATTN_EXT":
        return "Attention QK/softmax/AV"
    if o == "SSM_CONV":
        return "Convolution"
    if o in ["GET_ROWS", "SET_ROWS", "CPY", "CONT", "CONCAT", "DUP"]:
        return "Copies/gather/state movement"
    return "Vector/norm/activation/other"


def aggregate(results_dir, experiment):
    by_phase = collections.defaultdict(collections.Counter)
    op_phase = collections.defaultdict(collections.Counter)
    shapes, weights = {}, {}
    coverage = collections.defaultdict(collections.Counter)
    graphs = collections.defaultdict(set)
    dense_mac = collections.defaultdict(int)
    fused = records = 0
    for name in experiment["policy"]["op_trace_files"]:
        path = results_dir / name
        if not exists(path):
            continue
        for line in _lines(path):
            x = json.loads(line)
            m = re.fullmatch(r"p(\d+)_r(\d+)_(prefill|decode\d+)", x["phase"])
            if not m:
                raise ValueError("Unexpected phase " + x["phase"])
            if x["graph_status"] != 0 or x["elapsed_us"] < 0:
                raise ValueError("Failed graph/timer")
            n, rep, step = int(m[1]), int(m[2]), m[3]
            phase = "prefill" if step == "prefill" else "decode"
            key = (n, rep, phase)
            g = group(x)
            by_phase[key][g] += x["elapsed_us"]
            op_phase[key][x["tensor"]["op"]] += x["elapsed_us"]
            graphs[key].add(x["graph_id"])
            records += 1
            fused += bool(x["fused_successors"])
            if rep == 0 and step in ["prefill", "decode0"]:
                t, s = x["tensor"], x["src"]
                sk = (n, phase, t["name"], t["op"], tuple(t["ne"]),
                      tuple((z["name"], z["dtype"], tuple(z["ne"])) for z in s if z))
                if sk not in shapes:
                    shapes[sk] = {"prompt_tokens": n, "phase": phase, "group": g, "tensor": t,
                                  "sources": [z for z in s if z], "fused_successors": x["fused_successors"],
                                  "calls": 0, "total_us": 0}
                shapes[sk]["calls"] += 1
                shapes[sk]["total_us"] += x["elapsed_us"]
                coverage[(n, phase)][t["op"]] += 1
                if t["op"] == "MUL_MAT":
                    dense_mac[(n, phase, g)] += math.prod(t["ne"]) * s[0]["ne"][0]
                    if ".weight" in s[0]["name"]:
                        weights[s[0]["name"]] = {k: s[0][k] for k in ["name", "dtype", "ne", "nbytes"]}
    if not records:
        raise ValueError("No CPU operation records found in " + str(results_dir))
    rows = []
    for (n, rep, ph), c in sorted(by_phase.items()):
        total = sum(c.values())
        eligible = timing_eligible(experiment, n)
        rows.append({
            "prompt_tokens": n, "rep": rep, "phase": ph, "timing_eligible_for_report": eligible,
            "timing_note": "CPU interval shares; not baseline throughput" if eligible
                           else experiment["policy"]["timing_exclusion_note"],
            "timed_interval_us": total, "graphs": len(graphs[(n, rep, ph)]),
            "groups_us": dict(c), "groups_percent": {k: v / total * 100 for k, v in c.items()},
            "operations_us": dict(op_phase[(n, rep, ph)])})
    summary = {
        "records": records, "fused_records": fused, "runs": rows,
        "mac_counts": [{"prompt_tokens": n, "phase": ph, "group": g, "macs": v}
                       for (n, ph, g), v in dense_mac.items()],
        "coverage": [{"prompt_tokens": n, "phase": ph, "op_calls": dict(v)}
                     for (n, ph), v in coverage.items()]}
    return summary, list(shapes.values()), list(weights.values())


def write_outputs(out_dir, summary, shapes, weights):
    write_json(out_dir / "operation-summary.json", summary)
    write_json(out_dir / "operator-shapes.json", shapes)
    write_json(out_dir / "matrix-weight-inventory.json", weights)
    with open_csv(out_dir / "operation-summary.csv") as stream:
        w = csv.writer(stream)
        w.writerow(["prompt_tokens", "rep", "phase", "group", "time_us",
                    "percent_of_timed_CPU_intervals", "timing_eligible_for_report"])
        for r in summary["runs"]:
            for g, v in r["groups_us"].items():
                w.writerow([r["prompt_tokens"], r["rep"], r["phase"], g, v,
                            r["groups_percent"][g], r["timing_eligible_for_report"]])


def main(argv=None):
    args = parse_paths(__doc__, argv=argv)
    summary, shapes, weights = aggregate(args.results_dir, args.experiment)
    write_outputs(args.output_dir, summary, shapes, weights)
    for r in summary["runs"]:
        print(r["prompt_tokens"], r["rep"], r["phase"], round(r["timed_interval_us"] / 1000, 2),
              {g: round(p, 1) for g, p in r["groups_percent"].items()})
    print("records", summary["records"], "fused", summary["fused_records"])
    print("Output directory:", args.output_dir)


if __name__ == "__main__":
    main()
