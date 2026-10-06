"""Build results/index.html: one self-contained page over a run's captured graphs, with graph diffing.

Reads <run>/graphs/*/{prefill,decode}.{summary.json,ops.csv,*.fusion-chains.json,...} and the run's analysis/ and
research/ outputs; nothing here runs the model. SVG/PNG figures are linked by relative path, so keep index.html
next to the run folders. Op sequences are embedded (string-interned) so the diff works from file:// with no server.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import datasets
from result_io import load_json, write_text

PHASES = ("prefill", "decode")
GRAPH_RE = re.compile(r"^(?:prompt-(?P<prompt>.+)|pp(?P<length>\d+))-fa-(?P<fa>on|off)$")
TEMPLATE = Path(__file__).with_name("site_template.html")


class Strings:
    """Interns repeated strings so 40+ op sequences of ~1.4k rows stay small in the page."""

    def __init__(self):
        self.items, self.index = [], {}

    def __call__(self, value):
        if value not in self.index:
            self.index[value] = len(self.items)
            self.items.append(value)
        return self.index[value]


def latest_run(results_dir):
    runs = []
    for manifest in Path(results_dir).glob("*/run-manifest.json"):
        if not (manifest.parent / "graphs").is_dir():
            continue
        started = json.loads(manifest.read_text(encoding="utf-8")).get("run", {}).get("started_at", "")
        runs.append((started, manifest.parent))
    if not runs:
        raise SystemExit(f"No run with graphs/ under {results_dir}")
    return max(runs)[1]


def read_json(path):
    return load_json(path) if Path(path).is_file() else None


def phase_data(directory, phase, strings, rel, raw_links=True):
    summary = load_json(directory / f"{phase}.summary.json")
    with (directory / f"{phase}.ops.csv").open(encoding="utf-8", newline="") as stream:
        rows = [[strings(r["op"]), strings(r["category"]), int(r["layer"]), strings(r["dtype"]), strings(r["shape"]),
                 int(r["macs"]) if r["macs"] else 0, int(r["logical_output_bytes"]), strings(r["name"])]
                for r in csv.DictReader(stream)]
    ops_hash = hashlib.sha1("|".join(str(r[0]) for r in rows).encode()).hexdigest()[:12]
    shape_hash = hashlib.sha1("|".join(f"{r[0]}.{r[3]}.{r[4]}" for r in rows).encode()).hexdigest()[:12]

    chains = read_json(directory / f"{phase}.fusion-chains.json")
    fusion = None
    if chains is not None:
        signatures = {}
        for chain in chains:
            row = signatures.setdefault(" > ".join(chain["op_sequence"]), [0, 0, 0, 0, 0])
            row[0] += 1
            row[1] += chain["matrix_macs"]
            row[2] += chain["materialization_bytes_if_kept_separate"]
            backend = chain.get("backend_fusion") or {}  # absent when the graph has no paired op trace
            row[3] += backend.get("edges_fused", 0)
            row[4] += backend.get("edges_total", 0)
        fusion = {"chains": len(chains), "ops": sum(c["length"] for c in chains),
                  "signatures": [[sig, *vals] for sig, vals in sorted(signatures.items())]}

    svgs = {}
    for layer in ("layer0", "layer3"):
        if (directory / f"{phase}.{layer}.svg").is_file():
            svgs[layer.removeprefix("layer")] = f"{rel}/{phase}.{layer}.svg"
    if (directory / f"{phase}.svg").is_file():
        svgs["full"] = f"{rel}/{phase}.svg"
    files = {label: f"{rel}/{phase}.{name}" for label, name in
             [("raw graph", "json"), ("full DOT", "dot"), ("operations", "ops.csv"), ("summary", "summary.json"),
              ("fusion chains", "fusion-chains.json"), ("category transitions", "category-transitions.json"),
              ("offload boundaries", "offload-boundaries.json")] if raw_links and (directory / f"{phase}.{name}").is_file()}
    meta = summary["metadata"]
    return {
        "input_tokens": meta["input_tokens"], "past_tokens": meta["past_tokens"], "nodes": summary["scheduled_nodes"],
        "cats": summary["category_counts"], "ops": summary["operation_counts"], "macs": summary["macs_by_supported_op"],
        "valid": all(summary["validation"].values()), "limitations": summary["limitations"],
        "seq_hash": ops_hash, "shape_hash": shape_hash, "rows": rows,
        "offload": read_json(directory / f"{phase}.offload-boundaries.json"),
        "transitions": read_json(directory / f"{phase}.category-transitions.json"),
        "fusion": fusion, "svgs": svgs, "files": files,
    }


def graph_entry(directory, run_rel, strings, raw_links=True):
    match = GRAPH_RE.match(directory.name)
    rel = f"{run_rel}/graphs/{directory.name}"
    phases = {p: phase_data(directory, p, strings, rel, raw_links) for p in PHASES}
    meta = load_json(directory / "prefill.summary.json")["metadata"]
    fa = match["fa"]
    if match["prompt"]:
        label = f"prompt {match['prompt']} ({meta['prompt_tokens']} tok) · FA {fa}"
    else:
        label = f"pp{match['length']} · FA {fa}"
    reports = {name: f"{rel}/{name}" for name in ("REPORT.md", "FUSION-REPORT.md", "OFFLOAD-REPORT.md")
               if raw_links and (directory / name).is_file()}
    return {"id": directory.name, "label": label, "prompt": match["prompt"], "fa": fa,
            "tokens": meta["prompt_tokens"], "context": meta["context_capacity"], "reports": reports, "phases": phases}


def order_key(graph):
    return (graph["prompt"] is not None, graph["tokens"], graph["prompt"] or "", graph["fa"] != "on")


def performance_rows(run_dir, dataset_rows):
    path = run_dir / "analysis" / "summary.csv"
    if not path.is_file():
        return []
    keep = ("run", "prompt_tokens", "prefill_tps", "decode_ms", "decode_tps")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [{k: row[k] for k in keep} for row in csv.DictReader(stream)]
    for row in rows:
        dataset = datasets.for_run(dataset_rows, row["run"])
        row["dataset_role"] = dataset.get("role") if dataset else "unknown"
        row["dataset_label"] = datasets.label(dataset)
        row["build"] = dataset.get("build") if dataset else "unknown"
    return rows


def build(run_dir, out_path, raw_links=True):
    run_dir = Path(run_dir).resolve()
    run_rel = run_dir.name
    strings = Strings()
    graphs = [graph_entry(d, run_rel, strings, raw_links) for d in sorted((run_dir / "graphs").glob("*"))
              if GRAPH_RE.match(d.name) and (d / "prefill.summary.json").is_file() and (d / "decode.summary.json").is_file()]
    graphs.sort(key=order_key)
    if not graphs:
        raise ValueError("No derived graph summaries found; refusing to write an empty graph dashboard. "
                         "Restore the omitted graph analysis files before rebuilding this page.")
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    dataset_rows = datasets.read(run_dir)
    research = run_dir / "research" / "compute-mapping.md"
    figures = {name: f"{run_rel}/analysis/{name}" for name in ("profiling-summary.png", "decode-curve.png")
               if (run_dir / "analysis" / name).is_file()}
    data = {
        "run": {"name": run_rel, **manifest["run"]}, "host": manifest["host"], "config": manifest["config"],
        "comparability": manifest.get("comparability"), "subtitle": manifest.get("plot_subtitle"),
        "performance": performance_rows(run_dir, dataset_rows), "figures": figures,
        "datasets": dataset_rows,
        "builds": {"baseline": manifest.get("build"), "cuda": manifest.get("cuda_build"),
                   "profile": manifest.get("profile_build")},
        "prompt_comparison": (run_dir / "graphs" / "prompt-comparison.md").read_text(encoding="utf-8")
        if (run_dir / "graphs" / "prompt-comparison.md").is_file() else None,
        "research": research.read_text(encoding="utf-8") if research.is_file() else None,
        "strings": strings.items, "graphs": graphs,
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("__TITLE__", run_rel).replace("__DATA__", payload)
    write_text(out_path, page)
    return len(graphs), len(page)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, nargs="?", help="results/<run>; default: the newest run with graphs/")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--output", type=Path, help="Default: <results-dir>/index.html")
    parser.add_argument("--no-raw-links", action="store_true",
                        help="Link only SVG/PNG figures, not per-graph JSON/CSV/DOT/Markdown files (for the committed page, "
                             "where those raw files are not in the repository)")
    args = parser.parse_args(argv)
    run_dir = args.run_dir or latest_run(args.results_dir)
    out = args.output or args.results_dir / "index.html"
    count, size = build(run_dir, out, raw_links=not args.no_raw_links)
    print(f"Wrote {out} ({size / 1e6:.1f} MB): {count} graphs from {Path(run_dir).name}")


if __name__ == "__main__":
    main()
