"""Operation-sharing evidence: structurally-fusable chains and category-boundary crossings.

Reads one captured graph (prefill.json/decode.json, the same files graph.py analyzes) and,
optionally, the CPU operation trace of a matching run so real backend fusion (ggml's
`fused_successors`) can be checked against what the dataflow alone makes fusable.

Two dataflow facts are computed, both derived only from the schedule already captured
(no new instrumentation, no re-running the model):

- A *chain* is a maximal run of scheduled tensors t1 -> t2 -> ... -> tk where each t(i+1)'s
  only computed input is t(i), and t(i) has no other computed consumer. That is the same
  "safe to keep in one pipeline stage without spilling the intermediate" criterion used by
  elementwise kernel fusion; it says nothing about what a specific backend or accelerator
  actually fuses.
- A *category transition* is every scheduled edge (producer -> consumer), tagged "chainable"
  when it meets the chain criterion above and "shared" otherwise (the producer feeds more than
  one consumer, or the consumer merges more than one computed input). Shared edges are boundary
  crossings no fusion choice can remove; chainable edges are candidates.

Backend-fusion cross-check, categories and boundary-byte costs are all read from the same
tensor records graph.py already validates; nothing here re-derives timing or hardware cost.
"""
import argparse
import collections
import json
from pathlib import Path
import re

from graph import category, macs_of
from result_io import exists, open_result, output_dir, write_json, write_text


def _load_phase(path):
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    tensors = {t["id"]: t for t in graph["tensors"]}
    order = graph["scheduled_nodes"]
    return graph, tensors, order


def computed_predecessors(t, pos):
    """Sources that are themselves scheduled nodes (not a weight/input leaf)."""
    return [s["tensor"] for s in t["sources"] if s["tensor"] in pos]


def extract_chains(tensors, order):
    """Return (chains, fanout, pos). fanout counts scheduled consumers per scheduled producer."""
    pos = {i: idx for idx, i in enumerate(order)}
    fanout = collections.Counter()
    for i in order:
        for p in computed_predecessors(tensors[i], pos):
            fanout[p] += 1
    chain_of, chains = {}, []
    for i in order:
        preds = computed_predecessors(tensors[i], pos)
        parent = preds[0] if len(preds) == 1 else None
        if parent is not None and fanout[parent] == 1 and parent in chain_of:
            chain = chain_of[parent]
            chain.append(i)
        else:
            chain = [i]
            chains.append(chain)
        chain_of[i] = chain
    return chains, fanout, pos


PHASE_RE = re.compile(r"p(\d+)_r(\d+)_(prefill|decode\d+)")


def load_fusion_lookup(op_trace_path, phase, prompt_tokens=None):
    """{producer tensor name: [names ggml fused right after it]} from repetition 0 of one run.

    phase: 'prefill' or 'decode' (matched against the trace's first decode step, decode0).
    prompt_tokens: required when op_trace_path may cover more than one prompt length (the
    fixed-length trace mixes 128/512/2048 in one file); a per-prompt trace has only one length."""
    if op_trace_path is None or not exists(op_trace_path):
        return None
    want_step = "prefill" if phase == "prefill" else "decode0"
    lookup = {}
    found = False
    with open_result(op_trace_path) as stream:
        for line in stream:
            x = json.loads(line)
            m = PHASE_RE.fullmatch(x["phase"])
            if not m or int(m[2]) != 0 or m[3] != want_step:
                continue
            if prompt_tokens is not None and int(m[1]) != prompt_tokens:
                continue
            found = True
            lookup[x["tensor"]["name"]] = [s["name"] for s in x["fused_successors"] if s]
    return lookup if found else None


def analyze_phase(path, op_trace_path, phase):
    graph, tensors, order = _load_phase(path)
    fusion_lookup = load_fusion_lookup(op_trace_path, phase, graph["metadata"]["prompt_tokens"]) \
        if op_trace_path else None
    chains, fanout, pos = extract_chains(tensors, order)
    chain_rows = []
    transitions = collections.Counter()       # (from_cat, to_cat, kind) -> edge count
    transition_bytes = collections.Counter()  # (from_cat, to_cat, kind) -> producer bytes
    for chain in chains:
        cats = [category(tensors[i]) for i in chain]
        macs_total = sum(m for i in chain if (m := macs_of(tensors[i], tensors)) is not None)
        head = tensors[chain[0]]
        tail_id = chain[-1]
        weight_bytes = sum(tensors[s["tensor"]]["logical_bytes"] for i in chain
                            for s in tensors[i]["sources"] if s["tensor"] not in pos)
        external_in_bytes = sum(tensors[p]["logical_bytes"] for p in computed_predecessors(head, pos))
        tail_fanout = fanout.get(tail_id, 0)
        row = {
            "chain_id": len(chain_rows), "length": len(chain),
            "op_sequence": [tensors[i]["op_desc"] for i in chain],
            "categories": cats,
            "matrix_macs": macs_total,
            "external_activation_input_bytes": external_in_bytes,
            "weight_input_bytes": weight_bytes,
            "materialization_bytes_if_kept_separate": tensors[tail_id]["logical_bytes"] if tail_fanout > 0 else 0,
            "is_graph_output": tail_fanout == 0,
        }
        if fusion_lookup is not None:
            fused = sum(1 for a, b in zip(chain, chain[1:])
                        if tensors[b]["name"] in fusion_lookup.get(tensors[a]["name"], []))
            row["backend_fusion"] = {"edges_fused": fused, "edges_total": len(chain) - 1}
        chain_rows.append(row)
    for i in order:
        t = tensors[i]
        preds = computed_predecessors(t, pos)
        kind = "chainable" if len(preds) == 1 and fanout[preds[0]] == 1 else "shared"
        for p in preds:
            pair = (category(tensors[p]), category(t), kind)
            transitions[pair] += 1
            transition_bytes[pair] += tensors[p]["logical_bytes"]
    transition_rows = sorted(
        ({"from_category": f, "to_category": t, "edge_kind": k, "edges": n,
          "producer_bytes": transition_bytes[(f, t, k)]} for (f, t, k), n in transitions.items()),
        key=lambda r: -r["producer_bytes"])
    return chain_rows, transition_rows


def _report(graph_dir, phases):
    lines = ["# Operation-sharing evidence: chains and category boundaries", "",
             f"Workload: {graph_dir.name}.", "",
             "A **chain** is a maximal run of ops where each step's only computed input is the "
             "previous step's output, and that output has no other consumer -- the dataflow "
             "condition needed to keep an intermediate out of memory, not a claim about what any "
             "specific backend or accelerator fuses. A **category transition** is every scheduled "
             "producer-consumer edge, tagged `chainable` (meets that condition) or `shared` "
             "(the producer feeds more than one consumer, or the consumer merges several computed "
             "inputs, so materializing that edge cannot be avoided by fusion choices alone).", ""]
    for phase, (chains, transitions) in phases.items():
        lines += [f"## {phase}", "",
                  f"{len(chains)} chains covering {sum(c['length'] for c in chains)} scheduled ops.", ""]
        top = sorted(chains, key=lambda c: -c["matrix_macs"])[:10]
        lines += ["### Highest-MAC chains", "",
                  "| Length | Matrix MACs | Categories | Backend-fused edges |",
                  "|---:|---:|---|---|"]
        for c in top:
            fused = f"{c['backend_fusion']['edges_fused']}/{c['backend_fusion']['edges_total']}" \
                if c.get("backend_fusion") else "not checked"
            lines.append(f"| {c['length']} | {c['matrix_macs']:,} | "
                         f"{' -> '.join(dict.fromkeys(c['categories']))} | {fused} |")
        lines += ["", "### Category-boundary crossings (top by producer bytes)", "",
                  "| From | To | Kind | Edges | Producer bytes |", "|---|---|---|---:|---:|"]
        for r in transitions[:15]:
            lines.append(f"| {r['from_category']} | {r['to_category']} | {r['edge_kind']} | "
                         f"{r['edges']} | {r['producer_bytes']:,} |")
        lines.append("")
    lines += ["## Limitations", "",
              "- Chains and transitions come from one captured graph per phase; a different prompt "
              "length keeps the same shapes-independent structure (see `graphs/prompt-comparison.md`) "
              "but was not re-checked here.",
              "- `backend_fusion` reflects ggml's CPU scheduler on this host; a target accelerator's "
              "backend may fuse a different set of adjacent ops.",
              "- Bytes are logical tensor sizes (`logical_bytes`), not measured DRAM traffic; views "
              "alias storage and kernels may reuse buffers the byte count does not know about.",
              "- \"Shared\" edges are a lower bound on real materialization: dependency-graph fan-out "
              "does not capture scheduler reordering or buffer-lifetime overlap.", ""]
    write_text(graph_dir / "FUSION-REPORT.md", "\n".join(lines))


def analyze_directory(graph_dir, op_trace_path=None, out_dir=None):
    graph_dir = Path(graph_dir).resolve()
    out_dir = output_dir(Path(out_dir)) if out_dir else output_dir(graph_dir)
    phases = {}
    for phase in ("prefill", "decode"):
        chains, transitions = analyze_phase(graph_dir / f"{phase}.json", op_trace_path, phase)
        write_json(out_dir / f"{phase}.fusion-chains.json", chains, trailing_newline=True)
        write_json(out_dir / f"{phase}.category-transitions.json", transitions, trailing_newline=True)
        phases[phase] = (chains, transitions)
    _report(out_dir, phases)
    return phases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("graph_dir", type=Path, help="Directory holding prefill.json and decode.json")
    parser.add_argument("--op-trace", type=Path,
                        help="cpu-op-trace.jsonl(.gz) from the run that captured this graph, "
                             "for real backend-fusion cross-check")
    parser.add_argument("--output-dir", type=Path, help="Default: the graph directory itself")
    args = parser.parse_args(argv)
    phases = analyze_directory(args.graph_dir, args.op_trace, args.output_dir)
    print(json.dumps({p: {"chains": len(c), "transitions": len(t)} for p, (c, t) in phases.items()}, indent=2))


if __name__ == "__main__":
    main()
