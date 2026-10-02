"""Validate captured GGML graphs and derive hardware-independent work inventories.

Ported from Zeb Taylor's scripts/analyze_graph.py (experiments/llama-cpp/2026-09-24-qwen35-2b,
branch import/qwen35-2b-profiling-2026-09-24, commit 2835060). Validation, work counting and
the DOT/CSV/summary formats are unchanged. Generalized: outputs go to a chosen directory,
example layers are found from the graph instead of assumed, Graphviz is optional, and the
HTML index text is derived from the data.
"""
import argparse
import collections
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from result_io import open_csv, output_dir, write_json, write_text

VIEWS = {"NONE", "VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"}
COLORS = {"matrix": "#dbeafe", "recurrent": "#ede9fe", "convolution": "#ccfbf1",
          "vector": "#fef3c7", "memory": "#e2e8f0", "metadata": "#f1f5f9"}
RECURRENT_OPS = {"GATED_DELTA_NET"}
ATTENTION_OPS = {"SOFT_MAX", "FLASH_ATTN_EXT"}


def category(n):
    op = n["op"]
    if op in VIEWS:
        return "metadata"
    if op in {"MUL_MAT", "MUL_MAT_ID"}:
        return "matrix"
    if op == "GATED_DELTA_NET":
        return "recurrent"
    if op == "SSM_CONV":
        return "convolution"
    if op in {"CPY", "CONT", "SET_ROWS", "GET_ROWS", "CONCAT"}:
        return "memory"
    return "vector"


def example_layers(order, ts, layers):
    """First layer containing recurrence and first containing attention (whichever exist)."""
    by_layer = collections.defaultdict(set)
    for i in order:
        by_layer[layers[i]].add(ts[i]["op"])
    chosen = []
    for wanted, label in ((RECURRENT_OPS, "recurrent"), (ATTENTION_OPS, "full attention")):
        found = sorted(k for k, ops in by_layer.items() if k >= 0 and ops & wanted)
        if found:
            chosen.append((found[0], label))
    return chosen or [(min((k for k in by_layer if k >= 0), default=0), "first layer")]


def process(path, out_dir, have_dot):
    graph = json.loads(path.read_text(encoding="utf-8"))
    ts = {t["id"]: t for t in graph["tensors"]}
    order = graph["scheduled_nodes"]
    pos = {n: i for i, n in enumerate(order)}
    assert len(ts) == len(graph["tensors"]) and len(set(order)) == len(order)
    assert graph["metadata"]["finite_logits"]
    visiting, done = set(), set()

    def validate(i):
        assert i in ts and i not in visiting, ("missing tensor or cycle", i)
        if i in done:
            return
        visiting.add(i)
        t = ts[i]
        assert t["elements"] == math.prod(t["shape"])
        for s in t["sources"]:
            validate(s["tensor"])
        if t["view_source"] >= 0:
            validate(t["view_source"])
        visiting.remove(i)
        done.add(i)

    for i in ts:
        validate(i)
    for i in order:
        for s in ts[i]["sources"]:
            j = s["tensor"]
            if j in pos:
                assert pos[j] < pos[i], ("non-topological scheduled dependency", i, j)
    layers = {}

    def layer(i):
        if i in layers:
            return layers[i]
        t = ts[i]
        name = t["name"]
        match = re.search(r"^blk\.(\d+)\.", name) or re.search(r"-(\d+)(?:\s|$)", name)
        value = int(match.group(1)) if match else max((layer(s["tensor"]) for s in t["sources"]), default=-1)
        layers[i] = value
        return value

    for i in ts:
        layer(i)
    rows, counts, work, cats, unresolved = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for index, i in enumerate(order):
        t = ts[i]
        op = t["op"]
        counts[t["op_desc"]] += 1
        cats[category(t)] += 1
        macs = None
        if op == "MUL_MAT":
            macs = t["elements"] * ts[t["sources"][0]["tensor"]]["shape"][0]
        elif op == "SSM_CONV":
            macs = t["elements"] * ts[t["sources"][1]["tensor"]]["shape"][0]
        if macs is not None:
            work[op] += macs
        elif category(t) not in {"metadata", "memory"}:
            unresolved[t["op_desc"]] += 1
        rows.append({"index": index, "tensor_id": i, "name": t["name"], "op": t["op_desc"],
                     "category": category(t), "layer": layers[i], "dtype": t["dtype"],
                     "shape": "x".join(map(str, t["shape"])), "logical_output_bytes": t["logical_bytes"],
                     "macs": macs if macs is not None else "",
                     "arithmetic_ops_2_per_mac": 2 * macs if macs is not None else "",
                     "source_tensor_ids": ";".join(str(s["tensor"]) for s in t["sources"]),
                     "view_source": t["view_source"]})
    stem = path.stem
    with open_csv(out_dir / f"{stem}.ops.csv") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "metadata": graph["metadata"], "scheduled_nodes": len(order), "dependency_tensors": len(ts),
        "compute_and_memory_nodes": sum(v for k, v in cats.items() if k != "metadata"),
        "category_counts": dict(cats), "operation_counts": dict(counts),
        "macs_by_supported_op": dict(work), "arithmetic_ops_by_supported_op": {k: 2 * v for k, v in work.items()},
        "arithmetic_not_yet_modeled": dict(unresolved),
        "limitations": [
            "MAC counts cover MUL_MAT and SSM_CONV only; they are not total model FLOPs.",
            "Vector functions, reductions and fused GATED_DELTA_NET need separate hardware cost models.",
            "Logical tensor bytes and spans are not DRAM traffic; views alias storage and kernels reuse data.",
            "Dependencies are tensor dataflow. Stateful aliases and backend fusion require additional scheduling constraints.",
            f"CPU-selected GGML graph with flash attention {'enabled' if graph['metadata'].get('flash_attention') else 'disabled'}; "
            "a target accelerator may lower or fuse it differently."],
        "validation": {"acyclic": True, "source_ids_valid": True, "schedule_topological": True, "finite_logits": True}}
    write_text(out_dir / f"{stem}.summary.json", json.dumps(summary, indent=2) + "\n")

    def dot(selected, title):
        lines = ["digraph G {", "rankdir=TB;",
                 "graph [bgcolor=\"white\",fontname=\"sans\",label=" + json.dumps(title) + ",labelloc=t];",
                 "node [shape=box,style=\"rounded,filled\",fontname=\"sans\",fontsize=10];", "edge [color=\"#94a3b8\"];"]
        for i in sorted(selected):
            t = ts[i]
            label = f'{i}: {t["name"]}\n{t["op_desc"]} | {t["dtype"]} | ' + "x".join(map(str, t["shape"]))
            lines.append(f'n{i} [label={json.dumps(label)},fillcolor="{COLORS[category(t)]}"];')
        for i in sorted(selected):
            for s in ts[i]["sources"]:
                if s["tensor"] in selected:
                    lines.append(f'n{s["tensor"]} -> n{i} [label="{s["slot"]}"];')
            v = ts[i]["view_source"]
            if v in selected and v not in {s["tensor"] for s in ts[i]["sources"]}:
                lines.append(f"n{v} -> n{i} [style=dashed,label=\"alias\"];")
        lines.append("}")
        return "\n".join(lines) + "\n"

    write_text(out_dir / f"{stem}.dot", dot(set(ts), f"{path.parent.name} {stem}: complete tensor dataflow"))
    chosen = example_layers(order, ts, layers)
    svgs = []
    for il, label in chosen:
        selected = {i for i in order if layers[i] == il}
        selected.update(s["tensor"] for i in list(selected) for s in ts[i]["sources"])
        dest = out_dir / f"{stem}.layer{il}.dot"
        write_text(dest, dot(selected, f"{path.parent.name} {stem}: layer {il} (plus direct boundary inputs)"))
        if have_dot:
            subprocess.run(["dot", "-Tsvg", str(dest), "-o", str(dest.with_suffix(".svg"))], check=True, timeout=45)
            svgs.append((il, label))
    summary["_example_layers"] = chosen
    summary["_svg_layers"] = svgs
    return summary


def write_report(graph_dir, out_dir, summaries):
    report = ["# Qwen3.5-2B graph inventory", "",
              f"Workload: {graph_dir.name}. Graphs captured during successful CPU inference.", "",
              "| Phase | Input tokens | Past tokens | Scheduled nodes | Matrix MACs | Conv MACs |",
              "|---|---:|---:|---:|---:|---:|"]
    for phase, s in summaries.items():
        m, w = s["metadata"], s["macs_by_supported_op"]
        report.append(f'| {phase} | {m["input_tokens"]} | {m["past_tokens"]} | {s["scheduled_nodes"]:,} | '
                      f'{w.get("MUL_MAT", 0):,} | {w.get("SSM_CONV", 0):,} |')
    report += ["", "MAC = multiply-accumulate; the work inventory counts two arithmetic operations per MAC. "
               "These counts exclude vector/reduction work and fused recurrent operations. "
               "No accelerator latency or tokens/s is estimated yet.", "", "## Operation counts", "",
               "| Operation | Prefill | Decode |", "|---|---:|---:|"]
    pre, dec = summaries["prefill"]["operation_counts"], summaries["decode"]["operation_counts"]
    for op in pre:
        report.append(f"| {op} | {pre.get(op, 0)} | {dec.get(op, 0)} |")
    report += ["", "## Interpretation", ""] + ["- " + x for x in summaries["prefill"]["limitations"]]
    svg_layers = summaries["prefill"]["_svg_layers"]
    if svg_layers:
        shown = ("layer " + " and layer ".join(str(il) for il, _ in svg_layers) + " SVGs show representative "
                 + " and ".join(label.replace(" ", "-") for _, label in svg_layers) + " layers.")
    else:
        shown = "layer SVGs were not rendered because Graphviz (dot) was not available."
    report += ["", "Raw JSON contains all dependencies, data types, shapes, byte strides, alias offsets, operation "
               "parameters, and scheduler order. DOT files contain full dataflow; " + shown, ""]
    write_text(out_dir / "REPORT.md", "\n".join(report))


def write_index(graph_dir, out_dir, summaries):
    pre = summaries["prefill"]
    meta = pre["metadata"]
    table = "".join(f'<tr><td>{html.escape(op)}</td><td>{pre["operation_counts"].get(op, 0)}</td>'
                    f'<td>{summaries["decode"]["operation_counts"].get(op, 0)}</td></tr>' for op in pre["operation_counts"])
    links = "".join(f'<a href="{p}.{ext}">{p} {label}</a> ' for p in ["prefill", "decode"]
                    for ext, label in [("json", "raw graph"), ("dot", "full DOT"), ("ops.csv", "operations"), ("summary.json", "summary")])
    svg_layers = pre["_svg_layers"]
    facts = (f'{pre["scheduled_nodes"]:,} scheduled nodes per graph; flash attention '
             f'{"on" if meta.get("flash_attention") else "off"}; {meta.get("layers", "?")} layers.')
    text = ('<!doctype html><meta charset="utf-8"><title>Graph inventory</title><style>body{font:16px system-ui;color:#182b43;'
            'background:#f6f8fc;margin:30px auto;max-width:1200px;padding:0 24px}h1{font-size:32px}a{color:#155ab6;margin-right:16px;'
            'line-height:2.2}table{border-collapse:collapse;background:white}td,th{padding:7px 26px;text-align:left;border-bottom:1px solid #e2e8f0}'
            '.panel{background:white;padding:20px;border-radius:12px;margin:24px 0}.diagram{overflow:auto;height:650px;border:1px solid #dbe3ee;'
            'background:white}.diagram img{max-width:none;height:auto}select{padding:8px;font:inherit}.note{color:#52657d;line-height:1.6}</style>')
    text += f'<h1>{html.escape(graph_dir.name)}</h1><p>Captured llama.cpp computational graph · text inference · one sequence</p>'
    text += ('<div class="panel"><strong>Verified:</strong> successful prefill and decode with finite logits; tensor dependencies validated. '
             f'<p class="note">{html.escape(facts)}</p>') + links + "</div>"
    if svg_layers:
        options = "".join(f'<option value="{il}">Layer {il} · {label}</option>' for il, label in svg_layers)
        first = svg_layers[0][0]
        text += ('<div class="panel"><h2>Representative layer graphs</h2><p><select id="phase"><option value="prefill">Prefill</option>'
                 f'<option value="decode">Decode</option></select> <select id="layer">{options}</select> '
                 f'<a id="open" href="prefill.layer{first}.svg">Open full-size diagram</a></p><p class="note">Scroll to explore. '
                 'Blue: matrix; purple: recurrence; teal: convolution; yellow: vector; gray: memory/views. '
                 'Shapes use GGML innermost dimension first.</p><div class="diagram"><img id="diagram" '
                 f'src="prefill.layer{first}.svg" alt="Captured tensor dependency graph"></div></div>')
        text += ('<script>const phase=document.querySelector("#phase"),layer=document.querySelector("#layer");'
                 'function update(){let p=phase.value+".layer"+layer.value+".svg";document.querySelector("#diagram").src=p;'
                 'document.querySelector("#open").href=p}phase.onchange=update;layer.onchange=update;</script>')
    else:
        text += ('<div class="panel"><p class="note">Layer SVGs were not rendered because Graphviz (dot) was not available. '
                 'The DOT files above contain the same graphs; render them with any Graphviz installation.</p></div>')
    text += f'<div class="panel"><h2>Operation inventory</h2><table><tr><th>Operation</th><th>Prefill</th><th>Decode</th></tr>{table}</table></div>'
    text += '<p class="note">See REPORT.md for arithmetic coverage and limitations.</p>'
    write_text(out_dir / "index.html", text)


def analyze_directory(graph_dir, out_dir=None):
    """Analyze graph_dir/{prefill,decode}.json. Returns {phase: summary}."""
    graph_dir = Path(graph_dir).resolve()
    out_dir = output_dir(Path(out_dir)) if out_dir else output_dir(graph_dir)
    have_dot = shutil.which("dot") is not None
    summaries = {phase: process(graph_dir / f"{phase}.json", out_dir, have_dot) for phase in ("prefill", "decode")}
    write_report(graph_dir, out_dir, summaries)
    write_index(graph_dir, out_dir, summaries)
    return summaries, have_dot


PROMPT_DIR = re.compile(r"^prompt-(?P<id>.+)-fa-(?P<fa>on|off)$")


def fingerprints(directory):
    """Hashes of one captured graph (prefill then decode, in scheduled order).

    op_sequence: the operations only, i.e. the control flow. structure: operations plus data types and
    shapes, i.e. the whole dataflow graph. Tensor contents never enter a graph, so prompts with equal token
    counts give equal structure; different token counts change shapes but usually not the op sequence.
    """
    ops, shapes = hashlib.sha256(), hashlib.sha256()
    for phase in ("prefill", "decode"):
        data = json.loads((Path(directory) / f"{phase}.json").read_text(encoding="utf-8"))
        tensors = {t["id"]: t for t in data["tensors"]}
        for index in data["scheduled_nodes"]:
            t = tensors[index]
            ops.update(f"{phase}|{t['op_desc']}\n".encode())
            shapes.update(f"{phase}|{t['op_desc']}|{t['dtype']}|{t['shape']}\n".encode())
    return ops.hexdigest(), shapes.hexdigest()


def compare_prompt_graphs(run_dir):
    """Compare every graphs/prompt-<id>-fa-<on|off>/ folder of a run and write graphs/prompt-comparison.{json,md}.
    Returns the comparison dict, or None if the run has no per-prompt graphs."""
    graphs_dir = Path(run_dir) / "graphs"
    rows = []
    for directory in sorted(graphs_dir.glob("prompt-*")):
        match = PROMPT_DIR.match(directory.name)
        if not match or not (directory / "prefill.json").is_file():
            continue
        summaries = {p: json.loads((directory / f"{p}.summary.json").read_text(encoding="utf-8"))
                     for p in ("prefill", "decode")}
        op_hash, shape_hash = fingerprints(directory)
        meta = summaries["prefill"]["metadata"]
        rows.append({
            "prompt_id": match["id"], "flash_attention": match["fa"], "directory": directory.name,
            "prompt_tokens": meta["prompt_tokens"], "context_capacity": meta["context_capacity"],
            "scheduled_nodes": {p: s["scheduled_nodes"] for p, s in summaries.items()},
            "matrix_macs": {p: s["macs_by_supported_op"].get("MUL_MAT", 0) for p, s in summaries.items()},
            "op_sequence_sha256": op_hash, "structure_sha256": shape_hash})
    if not rows:
        return None
    result = {"prompts": rows, "groups": {}}
    lines = ["# Per-prompt graph comparison", "",
             "Each row is one captured prefill + single-decode-step graph. **Op sequence** is the control flow "
             "(which operations run, in order); **structure** also includes data types and shapes. "
             "Prompts of equal token count share a structure; prompts of different length usually keep the same "
             "op sequence with different shapes. Token contents never change a graph.", ""]
    for fa in ("on", "off"):
        subset = sorted((r for r in rows if r["flash_attention"] == fa), key=lambda r: (r["prompt_tokens"], r["prompt_id"]))
        if not subset:
            continue
        by_ops, by_shape = {}, {}
        for r in subset:
            by_ops.setdefault(r["op_sequence_sha256"], []).append(r["prompt_id"])
            by_shape.setdefault(r["structure_sha256"], []).append(r["prompt_id"])
        result["groups"][f"flash_attention_{fa}"] = {"op_sequence_groups": list(by_ops.values()),
                                                     "structure_groups": list(by_shape.values())}
        lines += [f"## Flash attention {fa}", "",
                  "| Prompt | Tokens | Context | Nodes (prefill/decode) | Matrix MACs prefill | Matrix MACs decode | Op sequence | Structure |",
                  "|---|---:|---:|---:|---:|---:|---|---|"]
        op_labels = {h: f"seq-{i + 1}" for i, h in enumerate(by_ops)}
        shape_labels = {h: f"shape-{i + 1}" for i, h in enumerate(by_shape)}
        for r in subset:
            lines.append(f"| {r['prompt_id']} | {r['prompt_tokens']} | {r['context_capacity']} | "
                         f"{r['scheduled_nodes']['prefill']}/{r['scheduled_nodes']['decode']} | "
                         f"{r['matrix_macs']['prefill']:,} | {r['matrix_macs']['decode']:,} | "
                         f"{op_labels[r['op_sequence_sha256']]} | {shape_labels[r['structure_sha256']]} |")
        lines += ["", f"Distinct op sequences: {len(by_ops)}. Distinct structures: {len(by_shape)}.", ""]
    write_json(graphs_dir / "prompt-comparison.json", result, trailing_newline=True)
    write_text(graphs_dir / "prompt-comparison.md", "\n".join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("graph_dir", type=Path, help="Directory holding prefill.json and decode.json")
    parser.add_argument("--output-dir", type=Path, help="Default: the graph directory itself")
    args = parser.parse_args(argv)
    summaries, have_dot = analyze_directory(args.graph_dir, args.output_dir)
    print(json.dumps({k: {"nodes": v["scheduled_nodes"], "matrix_macs": v["macs_by_supported_op"].get("MUL_MAT", 0),
                          "validated": v["validation"]} for k, v in summaries.items()}, indent=2))
    if not have_dot:
        print("Note: Graphviz 'dot' not found; DOT files written, SVGs skipped.")


if __name__ == "__main__":
    main()
