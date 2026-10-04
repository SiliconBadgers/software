"""Run fusion.py and offload.py across every captured graph in one results run, then synthesize
a cross-prompt operation-sharing and offload-boundary write-up into <run_dir>/research/.

Only reads already-captured graphs/*/{prefill,decode}.json and CPU op-trace files that a `profile`
run already produced; nothing here runs the model or re-derives timing.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import fusion
import offload
from result_io import exists, write_json, write_text

GRAPH_RE = re.compile(r"^(?:prompt-(?P<prompt>.+)|pp(?P<length>\d+))-fa-(?P<fa>on|off)$")


def op_trace_for(run_dir, name):
    """The fa-on graph matches the profiled configuration (see core/README.md); fa-off is
    structural only. Per-prompt graphs pair with that prompt's own trace; pp<N> graphs pair
    with the run's fixed-length trace (which fusion.py/offload.py filter to N)."""
    m = GRAPH_RE.match(name)
    if not m or m["fa"] != "on":
        return None
    path = (run_dir / "prompts" / m["prompt"] / "cpu-op-trace.jsonl.gz" if m["prompt"]
            else run_dir / "cpu-op-trace.jsonl.gz")
    return path if exists(path) else None


def run_all(run_dir):
    """Returns {graph_name: {"fusion": phases, "offload": phases, "op_trace": path or None}}."""
    results = {}
    for directory in sorted((run_dir / "graphs").glob("*")):
        if not (directory / "prefill.json").is_file() or not (directory / "decode.json").is_file():
            continue
        trace = op_trace_for(run_dir, directory.name)
        results[directory.name] = {
            "fusion": fusion.analyze_directory(directory, trace),
            "offload": offload.analyze_directory(directory, trace),
            # Run-relative, so the index is the same on every machine that regenerates it.
            "op_trace": trace.relative_to(run_dir).as_posix() if trace else None,
        }
    return results


def pick_longest(run_dir, results):
    """The largest-context per-prompt fa-on graph with a matching trace, for the synthesis
    tables; every prompt of a given flash-attention setting shares one op sequence
    (see graphs/prompt-comparison.md), so one representative's structure stands for all of them."""
    lengths = {}
    for name in results:
        m = GRAPH_RE.match(name)
        if m and m["prompt"] and name.endswith("-fa-on") and results[name]["op_trace"]:
            meta = json.loads((run_dir / "graphs" / name / "prefill.json").read_text(encoding="utf-8"))["metadata"]
            lengths[name] = meta["prompt_tokens"]
    if not lengths:
        return None
    return max(lengths, key=lengths.get)


def _offload_synthesis(results, representative):
    rows_by_phase = {"prefill": {}, "decode": {}}
    for phase in ("prefill", "decode"):
        for row in results[representative]["offload"][phase]:
            rows_by_phase[phase][row["group"]] = row
    lines = [f"### Offload boundaries ({representative}; structural bytes are prefill)", "",
             "| Group | Matrix MACs | Boundary bytes | Leaf-input bytes | Ops/boundary byte | Ops/(boundary+leaf) byte | "
             "Measured % prefill | Measured % decode |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    groups = sorted(rows_by_phase["prefill"], key=lambda g: -(rows_by_phase["prefill"][g].get("measured_cpu_time_percent") or 0))
    for g in groups:
        p, d = rows_by_phase["prefill"][g], rows_by_phase["decode"].get(g, {})
        aib = f"{p['arithmetic_ops_per_boundary_byte']:.1f}" if p.get("arithmetic_ops_per_boundary_byte") else "n/a"
        ail = f"{p['arithmetic_ops_per_boundary_and_leaf_byte']:.1f}" if p.get("arithmetic_ops_per_boundary_and_leaf_byte") else "n/a"
        lines.append(f"| {g} | {p['matrix_macs']:,} | {p['boundary_in_bytes'] + p['boundary_out_bytes']:,} | "
                     f"{p['leaf_input_bytes']:,} | {aib} | {ail} | "
                     f"{p.get('measured_cpu_time_percent', 0):.1f}% | {d.get('measured_cpu_time_percent', 0):.1f}% |")
    return "\n".join(lines)


def _fusion_synthesis(results, representative):
    lines = [f"### Category-boundary crossings ({representative}, prefill)", "",
             "| From | To | Kind | Producer bytes |", "|---|---|---|---:|"]
    for r in results[representative]["fusion"]["prefill"][1][:10]:
        lines.append(f"| {r['from_category']} | {r['to_category']} | {r['edge_kind']} | {r['producer_bytes']:,} |")
    return "\n".join(lines)


# offload.py maps only FLASH_ATTN_EXT to the attention group, so with flash attention off the QK and AV
# matmuls land in "Other matrix products" (and softmax in the vector group).
ATTENTION_GROUPS = ("Attention QK/softmax/AV", "Other matrix products")


def _mib(n):
    """Adaptive binary units, so an 8 KiB boundary does not print as 0.0 MiB."""
    if n >= 2**20:
        return f"{n / 2**20:,.1f} MiB"
    return f"{n / 2**10:,.1f} KiB" if n >= 2**10 else f"{n:,} B"


def _fixed(results, fa):
    """{prompt_tokens: graph name} for the fixed-length graphs of one flash-attention setting."""
    return {int(m["length"]): n for n in results if (m := GRAPH_RE.match(n)) and m["length"] and m["fa"] == fa}


def largest_intermediate(run_dir, name, phase):
    """Largest logical output of one scheduled op, excluding weights/inputs (NONE) and view-only ops
    (metadata category). One op's output size, not an allocator peak: a backend may alias or reuse buffers."""
    best = None
    with (run_dir / "graphs" / name / f"{phase}.ops.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["op"] == "NONE" or row["category"] == "metadata":
                continue
            size = int(row["logical_output_bytes"])
            if best is None or size > best[0]:
                best = (size, row["op"], row["name"], row["shape"])
    return best


def _boundary_bytes(rows):
    return {r["group"]: r["boundary_in_bytes"] + r["boundary_out_bytes"] for r in rows}


def _scaling_sections(run_dir, results):
    """Length and flash-attention evidence from the fixed-length graphs. Returns (lines, attention_note)."""
    on, off = _fixed(results, "on"), _fixed(results, "off")
    lengths = sorted(on)
    if len(lengths) < 2:
        return [], None
    lo, hi = lengths[0], lengths[-1]
    traced = [n for n in lengths if any(r.get("measured_cpu_time_percent") is not None for r in results[on[n]]["offload"]["prefill"])]
    lines = ["## Scaling with prompt length (fa-on, structural)", "",
             f"Fixed-length graphs at {', '.join(map(str, lengths))} tokens (context = 4x prompt). Boundary bytes are the logical size of "
             "every tensor crossing into or out of a group in one scheduled graph: an estimate of what an offload of that group "
             "must move, not a measured transfer.", ""]
    for phase in ("prefill", "decode"):
        per = {n: _boundary_bytes(results[on[n]]["offload"][phase]) for n in lengths}
        groups = sorted(per[hi], key=lambda g: -per[hi][g])
        lines += [f"### {phase.capitalize()} boundary bytes", "",
                  "| Group | " + " | ".join(f"pp{n}" for n in lengths) + f" | pp{hi} / pp{lo} |",
                  "|---|" + "---:|" * (len(lengths) + 1)]
        for g in groups:
            vals = [per[n].get(g, 0) for n in lengths]
            ratio = f"{vals[-1] / vals[0]:.1f}x" if vals[0] else "n/a"
            lines.append(f"| {g} | " + " | ".join(_mib(v) for v in vals) + f" | {ratio} |")
        lines.append("")
    if traced:
        untraced = [n for n in lengths if n not in traced]
        lines += [f"### Measured CPU share ({', '.join(f'pp{n}' for n in traced)} only)", "",
                  "Measured: CPU wall-clock interval shares from the instrumented run on this host.", ""]
        if untraced:
            lines += [f"{', '.join(f'pp{n}' for n in untraced)} has no measured share: the run's CPU operation trace has no records at "
                      "that length (the experiment excludes it from timing conclusions, and the traced run at that length is an opt-in "
                      "diagnostic that was not run), so it contributes structure only.", ""]
        lines += [
                  "| Group | " + " | ".join(f"{p} pp{n}" for p in ("prefill", "decode") for n in traced) + " |",
                  "|---|" + "---:|" * (2 * len(traced))]
        share = {(p, n): {r["group"]: r.get("measured_cpu_time_percent") or 0.0 for r in results[on[n]]["offload"][p]}
                 for p in ("prefill", "decode") for n in traced}
        for g in sorted(share[("prefill", traced[-1])], key=lambda g: -share[("prefill", traced[-1])][g]):
            lines.append(f"| {g} | " + " | ".join(f"{share[(p, n)].get(g, 0.0):.1f}%" for p in ("prefill", "decode") for n in traced) + " |")
        lines.append("")
    lines += ["## Largest single intermediate tensor", "",
              "Largest logical output of one scheduled op, excluding weights/inputs and view-only ops. It is one op's output, not an allocator peak; "
              "a backend may alias or reuse buffers, so read it as the size a unit must be able to hold or stream, not a memory budget.", "",
              "| Graph | Phase | Op | Tensor | Shape (ne0 innermost) | Size |", "|---|---|---|---|---|---:|"]
    for fa, table in (("on", on), ("off", off)):
        for n in sorted(table):
            for phase in ("prefill", "decode"):
                best = largest_intermediate(run_dir, table[n], phase)
                if best:
                    lines.append(f"| pp{n} fa-{fa} | {phase} | {best[1]} | `{best[2]}` | {best[3]} | {_mib(best[0])} |")
    lines.append("")
    note = None
    if hi in off:
        lines += [f"## Flash attention on vs off (pp{hi}, structural)", "",
                  "fa-on keeps attention inside one fused op; fa-off exposes QK, softmax and AV as separate scheduled ops whose intermediates cross unit boundaries.", "",
                  "| Phase | Nodes on / off | Matrix + conv MACs on / off | Attention-related boundary on / off | Sum over all groups on / off |", "|---|---|---|---|---|"]
        for phase in ("prefill", "decode"):
            rows = {fa: results[t[hi]]["offload"][phase] for fa, t in (("on", on), ("off", off))}
            crossing = lambda r: r["boundary_in_bytes"] + r["boundary_out_bytes"]
            att = {fa: sum(crossing(r) for r in rows[fa] if r["group"] in ATTENTION_GROUPS) for fa in rows}
            total = {fa: sum(crossing(r) for r in rows[fa]) for fa in rows}
            nodes = {fa: sum(r["nodes"] for r in rows[fa]) for fa in rows}
            macs = {fa: sum(r["matrix_macs"] for r in rows[fa]) for fa in rows}
            lines.append(f"| {phase} | {nodes['on']} / {nodes['off']} | {macs['on']:,} / {macs['off']:,} | "
                         f"{_mib(att['on'])} / {_mib(att['off'])} | {_mib(total['on'])} / {_mib(total['off'])} |")
            if phase == "prefill" and att["on"]:
                note = (f"At pp{hi} prefill the attention-related boundary is {_mib(att['on'])} with fa-on and {_mib(att['off'])} with fa-off "
                        f"({att['off'] / att['on']:.1f}x), and the whole graph's summed group boundaries grow from {_mib(total['on'])} to {_mib(total['off'])}; "
                        "whether attention stays fused inside one unit is therefore a boundary decision. This is structural evidence only (no measured timing at this length)")
        lines += ["", "*Attention-related* = the `Attention QK/softmax/AV` group plus `Other matrix products`. `offload.py` maps only `FLASH_ATTN_EXT` to the attention group, so with fa-off "
                  "the QK and AV matmuls are counted in `Other matrix products` and the softmax boundary is inside `Vector/norm/activation/other`; the fa-off figure is therefore a lower bound. "
                  "Sums add every group's in+out bytes, so a tensor crossing two groups is counted twice.", ""]
    return lines, note


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _configurations(run_dir, results):
    """Exactly what was run, read from run-manifest.json, analysis/summary.json and the graph folders
    present. A graph folder the manifest does not list was added after the run finished."""
    manifest = _load(run_dir / "run-manifest.json")
    lines = ["## Configurations in this evidence", ""]
    if manifest is None:
        return lines + ["No `run-manifest.json`: the configurations behind this write-up are unknown.", ""]
    run, host, cfg = manifest["run"], manifest["host"], manifest["config"]
    lines += [f"- Run `{run['name']}`, status `{run['status']}`, {run['started_at']} to {run['finished_at']}. "
              f"Command: `{run['command']}`.",
              f"- Host: {host['os']}, {host['cpu_model']}, {host['logical_cpus']} logical CPUs. Backends: "
              f"{', '.join(cfg['backends'])}; {cfg['threads']} threads; one sequence. No other OS or backend is in this run."]
    summary = _load(run_dir / "analysis" / "summary.json") or []
    for name in sorted({r["run"] for r in summary}):
        rows = [r for r in summary if r["run"] == name]
        lines.append(f"- Fixed-length `{name}` (measured): prompt tokens "
                     + ", ".join(f"{r['prompt_tokens']} ({r['repetitions']} reps x {r['decode_steps_per_rep']} decode steps)"
                                 for r in rows) + ".")
    timings = cfg.get("prompt_timings")
    profiles = _load(run_dir / "prompt-profiles.json")
    if timings and profiles:
        lines.append(f"- Per-prompt timings (measured): {len(profiles['prompts'])} prompts, untraced {timings['repetitions']} reps "
                     f"and traced {timings['trace_repetitions']} reps, {timings['decode_steps']} decode steps each.")
    missing = (manifest.get("analysis") or {}).get("missing_optional_diagnostic_profile") or []
    if missing:
        lines.append("- Not traced: " + ", ".join(f"{m['prompt_tokens']} {m['phase']}" for m in missing)
                     + " (optional diagnostic, not run). No measured operation share exists at that length.")
    listed = {Path(g["directory"]).name for g in manifest.get("graphs", []) + manifest.get("prompt_graphs", [])}
    in_run = sorted(n for n in results if n in listed)
    added = sorted(n for n in results if n not in listed)
    lines.append(f"- Graph captures recorded in the manifest ({len(in_run)}): {', '.join(in_run)}.")
    if added:
        lines.append(f"- **Graph captures added to this folder after the run finished, not recorded in `run-manifest.json`** "
                     f"({len(added)}): {', '.join(added)}. Each one's `prefill.json` metadata records its model file, threads, "
                     "context and flash-attention setting.")
    no_graph = [n for n in cfg["prompt_lengths"] if not any(GRAPH_RE.match(r)["length"] == str(n) for r in results)]
    if no_graph:
        lines.append(f"- Measured lengths with no captured graph: {', '.join(map(str, no_graph))}.")
    return lines + [""]


def _ranking_agreement(results, representative, k=3):
    """Compares each traced prompt's top-k groups by measured CPU share with the representative's."""
    def top(name, phase):
        rows = [r for r in results[name]["offload"][phase] if r.get("measured_cpu_time_percent") is not None]
        return [r["group"] for r in sorted(rows, key=lambda r: -r["measured_cpu_time_percent"])[:k]]
    traced = sorted(n for n in results
                    if (m := GRAPH_RE.match(n)) and m["prompt"] and m["fa"] == "on" and results[n]["op_trace"])
    lines = []
    for phase in ("prefill", "decode"):
        ref = top(representative, phase)
        differ = [n for n in traced if top(n, phase) != ref]
        lines.append(f"- {phase}: top-{k} groups by measured share ({', '.join(ref)}) match in "
                     f"{len(traced) - len(differ)} of {len(traced)} traced prompts"
                     + (f"; differ in {', '.join(differ)}" if differ else "") + ".")
    return lines


def _pct(row):
    return f"{row['measured_cpu_time_percent']:.1f}%" if row and row.get("measured_cpu_time_percent") is not None else "n/a"


def _observations(results, representative, attention_note):
    """Measured observations, then candidate boundaries with what each would still need."""
    def r(phase, group):
        return next((x for x in results[representative]["offload"][phase] if x["group"] == group), None)
    mlp, proj, vocab = "MLP projections", "Attention/DeltaNet projections", "Vocabulary output head"
    rec, conv, copies = "DeltaNet recurrence", "Convolution", "Copies/gather/state movement"
    vocab_d = r("decode", vocab)
    lines = ["## Measured CPU observations", "",
             f"From the CPU operation trace of {representative} (fa-on, this host only):", "",
             f"- {mlp} and {proj} take {_pct(r('prefill', mlp))} and {_pct(r('prefill', proj))} of prefill "
             f"and {_pct(r('decode', mlp))} and {_pct(r('decode', proj))} of decode operation time.",
             f"- The {vocab.lower()} takes {_pct(r('prefill', vocab))} of prefill and {_pct(vocab_d)} of decode.",
             f"- {copies} takes {_pct(r('prefill', copies))} of prefill and {_pct(r('decode', copies))} of decode.",
             f"- {rec} (`GATED_DELTA_NET`) takes {_pct(r('prefill', rec))} of prefill and {_pct(r('decode', rec))} of decode; "
             f"{conv} takes {_pct(r('prefill', conv))} and {_pct(r('decode', conv))}.", "",
             "These are shares of CPU kernel time under ggml's own CPU fusion and threading; they do not "
             "transfer to an accelerator's time split.", "",
             "## Candidate boundaries for discussion (not established)", "",
             "Each item states the evidence it rests on and what this run cannot show. None is a hardware "
             "recommendation; reconcile with the central architecture diagram and the Compute, Memory and Control "
             "owners before freezing a boundary.", "",
             f"1. **One shared matrix path for {mlp} and {proj}.** Measured: the largest CPU shares above. "
             "Structural: both are weight-times-activation MUL_MATs with the highest MAC counts. "
             "Not established: throughput on any accelerator, weight or activation SRAM traffic, or whether "
             "tiles fit any proposed buffer; that needs a memory/dataflow model or RTL."]
    if vocab_d:
        lines.append(f"2. **The {vocab.lower()} on the same matrix path.** Measured: {_pct(vocab_d)} of decode. Structural: "
                     f"its boundary is only {vocab_d['boundary_in_bytes'] + vocab_d['boundary_out_bytes']:,} bytes, but it reads "
                     f"{vocab_d['leaf_input_bytes']:,} bytes of weights per decode step "
                     f"({vocab_d['arithmetic_ops_per_boundary_and_leaf_byte'] or 0:.1f} ops per boundary+leaf byte), so in decode "
                     "it is dominated by weight reads, not arithmetic. Consistent with docs/profiling-next-steps.md. "
                     "Not established: weight bandwidth or latency on any target.")
    lines += [f"3. **Keep {copies.lower()} and vector/norm work next to their producers and consumers.** Measured: "
              "a noticeable decode share with no matrix arithmetic. Structural: large boundary bytes in the tables above. "
              "Not established: that a separate unit would stall, or what on-chip traffic either placement causes.",
              f"4. **{rec} and {conv}: a separate unit or mapping onto the matrix/vector units is still open.** "
              "Measured: small CPU shares. Structural: distinct op types in the graph (the per-graph fusion report lists their "
              "chains and crossings). Not established: that dedicated recurrence arithmetic is needed. CPU share and "
              "op identity cannot decide that; it needs the recurrence's cost on the proposed matrix/vector units."]
    if attention_note:
        lines.append(f"5. **Flash attention as a boundary choice.** {attention_note}.")
    return lines + [""]


def write_report(run_dir, results):
    out = run_dir / "research"
    out.mkdir(exist_ok=True)
    representative = pick_longest(run_dir, results)
    scaling, attention_note = _scaling_sections(run_dir, results)
    prompt_names = sorted(n for n in results if GRAPH_RE.match(n)["prompt"] and n.endswith("-fa-on"))
    fixed_names = sorted(n for n in results if GRAPH_RE.match(n)["length"])
    lines = ["# Operation-sharing and offload-boundary evidence", "",
             f"Run: `{run_dir.name}`. Generated by `core/analysis/compute_mapping.py` "
             "(fusion.py + offload.py) from already-captured graphs and CPU op traces; nothing "
             "here re-runs the model.", "",
             "Statements below are of three kinds:", "",
             "- **Measured**: CPU wall-clock interval shares from the instrumented llama.cpp CPU run on this host.",
             "- **Structural**: counts and logical tensor sizes read from the captured, CPU-scheduled GGML graph "
             "(MACs, boundary bytes, leaf-input bytes, chains). Not measured traffic and not an allocator trace.",
             "- **Candidate**: an interpretation for the boundary discussion. Not established by this run; "
             "each one lists what would be needed to establish it.", ""]
    lines += _configurations(run_dir, results)
    lines += ["## Coverage", "",
              f"- {len(prompt_names)} representative prompts x flash-attention on/off "
              f"({', '.join(n.removesuffix('-fa-on') for n in prompt_names)})",
              f"- {len(fixed_names)} fixed-length graphs x flash-attention on/off "
              f"({', '.join(sorted(set(n.rsplit('-fa-', 1)[0] for n in fixed_names)))})",
              "- Per-graph detail: the fusion and offload tables for every graph are shown in `results/index.html`; "
              "`graphs/<name>/FUSION-REPORT.md` and `OFFLOAD-REPORT.md` are regenerated locally by this script",
              "- fa-off graphs, and graphs whose length has no trace records, carry structure only", ""]
    if representative:
        lines += ["## Representative evidence", "",
                  "All prompts of a given flash-attention setting share one op sequence "
                  "(`graphs/prompt-comparison.md`), so the longest traced prompt, "
                  f"**{representative}**, is used below. Ranking check against the other traced prompts:", ""]
        lines += _ranking_agreement(results, representative) + [""]
        lines += [_offload_synthesis(results, representative), "",
                  "Matrix MACs, boundary bytes, leaf-input bytes and both ops/byte ratios are structural; only the "
                  "`Measured %` columns are measured.", "",
                  _fusion_synthesis(results, representative), ""]
        lines += scaling
        lines += _observations(results, representative, attention_note)
    lines += ["## What this evidence does not establish", "",
              "- SRAM or any other on-chip memory traffic, buffer sizing or bank conflicts.",
              "- FPGA or ASIC stalls, cycle counts, clock rates, area or power.",
              "- Transfer, dispatch or synchronization time across any proposed boundary. None is measured; "
              "`offload.py --bandwidth-gbps/--dispatch-us` only produce illustrative figures and were not used here.",
              "- The need for dedicated recurrence (DeltaNet) or convolution arithmetic.",
              "- Results for Linux, macOS, Metal, other CPUs or other thread counts, or model quality.", "",
              "## Limitations", "",
              "- Structural figures (boundary bytes, chains, ops/byte) come from the CPU-"
              "scheduled GGML graph on this host; a target accelerator's backend may schedule, fuse or "
              "buffer differently.",
              "- Measured percentages are CPU wall-clock interval shares (see run-manifest.json's "
              "comparability note), not accelerator area, bandwidth or cycle counts.",
              "- Largest-intermediate, boundary-byte and leaf-input figures are per-op logical sizes with no buffer reuse assumed; "
              "they are not allocator peaks or measured traffic, and no transfer, dispatch or synchronization time is measured here.",
              "- `Other matrix products` and any op not covered by ops.py's grouping fall outside this "
              "taxonomy; see the per-graph offload table in `results/index.html` (or a locally regenerated `OFFLOAD-REPORT.md`) "
              "for the full breakdown.",
              "- This is evidence for a boundary discussion, not a finished microarchitecture; it "
              "should be read alongside the central architecture diagram and reconciled with Compute, "
              "Memory and Control before any boundary is frozen.", ""]
    write_text(out / "compute-mapping.md", "\n".join(lines))
    write_json(out / "compute-mapping-index.json",
               {name: {"op_trace": r["op_trace"]} for name, r in results.items()}, trailing_newline=True)
    return out / "compute-mapping.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="A results/<run> directory with a graphs/ folder")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    results = run_all(run_dir)
    report = write_report(run_dir, results)
    print(f"Analyzed {len(results)} graphs. Write-up: {report}")


if __name__ == "__main__":
    main()
