"""One row per prompt: its timings, its operation trace, and the checks that go with them.

Reads results/<run>/prompts/<id>/ (written by the prompt-timing stage) and writes results/<run>/prompt-profiles.json.
Everything here is derived from the raw files; nothing is measured.
"""
import collections
import hashlib
import json
from pathlib import Path

import decode_curve
import datasets
import ops
import summarize
import validate
from policy import CORE, DEFAULT_EXPERIMENT, load_experiment
from result_io import exists, open_result, write_json

DEFAULT_PROMPT_SET = CORE / "prompts" / "prompts.json"
CONTINUATION = CORE / "prompts" / "continuation.txt"
TOP_OPERATIONS = 10


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metadata(path):
    with open_result(path) as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "metadata":
                return row
    raise ValueError(f"No metadata record in {path}")


def _percent(counter, total):
    return {key: 100 * value / total for key, value in sorted(counter.items(), key=lambda kv: -kv[1])}


def trace_block(summary, steps, baseline):
    """Aggregate a prompt's operation trace by phase; also compares it with the untraced timings."""
    phases = {}
    for phase in ("prefill", "decode"):
        runs = [r for r in summary["runs"] if r["phase"] == phase]
        total = sum(r["timed_interval_us"] for r in runs)
        groups, operations = collections.Counter(), collections.Counter()
        for r in runs:
            groups.update(r["groups_us"])
            operations.update(r["operations_us"])
        per_unit = total / len(runs) if phase == "prefill" else total / (len(runs) * steps)
        calls = next((c["op_calls"] for c in summary["coverage"] if c["phase"] == phase), {})
        macs = sum(m["macs"] for m in summary["mac_counts"] if m["phase"] == phase)
        phases[phase] = {
            "repetitions": len(runs),
            ("timed_interval_us_per_prefill" if phase == "prefill" else "timed_interval_us_per_token"): per_unit,
            "groups_percent": _percent(groups, total), "groups_us": dict(groups),
            "top_operations_percent": dict(list(_percent(operations, total).items())[:TOP_OPERATIONS]),
            "operation_calls": calls, "dense_matrix_macs": macs}
    prefill_s = phases["prefill"]["timed_interval_us_per_prefill"] / 1e6
    decode_ms = phases["decode"]["timed_interval_us_per_token"] / 1e3
    phases["profiler_overhead_check"] = {
        "note": "Sum of timed operation intervals divided by the untraced median. Near 1 means the instrumentation "
                "did not visibly slow execution; enabled-trace outer-call latency is never used.",
        "prefill_timed_over_untraced": prefill_s / baseline["prefill_seconds"],
        "decode_timed_over_untraced": decode_ms / baseline["decode_ms"]}
    phases["records"] = summary["records"]
    phases["fused_records"] = summary["fused_records"]
    return phases


def _timing(row, path=None, prompt_tokens=None):
    keep = ("repetitions", "decode_steps_per_rep", "prefill_seconds", "prefill_min_s", "prefill_max_s", "prefill_tps",
            "decode_ms", "decode_min_ms", "decode_max_ms", "decode_tps", "prefill_samples_seconds",
            "decode_run_means_ms")
    timing = {key: row[key] for key in keep}
    if path is not None:      # how per-step cost moves with context length; see decode_curve.py
        timing["decode_curve"] = decode_curve.compact(decode_curve.curves_for_file(path)[prompt_tokens])
    return timing


def _graphs(run_dir, prompt_id):
    comparison = run_dir / "graphs" / "prompt-comparison.json"
    by_setting = {}
    if comparison.is_file():
        for r in json.loads(comparison.read_text(encoding="utf-8"))["prompts"]:
            if r["prompt_id"] == prompt_id:
                by_setting[r["flash_attention"]] = r
    found = {}
    for fa in ("on", "off"):
        directory = run_dir / "graphs" / f"prompt-{prompt_id}-fa-{fa}"
        if directory.is_dir():
            entry = {"directory": f"graphs/{directory.name}"}
            if fa in by_setting:
                entry.update(structure_sha256=by_setting[fa]["structure_sha256"],
                             op_sequence_sha256=by_setting[fa]["op_sequence_sha256"],
                             scheduled_nodes=by_setting[fa]["scheduled_nodes"])
            found[f"flash_attention_{fa}"] = entry
    return found


def build_row(run_dir, experiment, entry, prompt_dir, prompt_set_dir):
    prompt_id = entry["id"]
    meta = _metadata(prompt_dir / "cpu-baseline.jsonl")
    n, steps = meta["prompt_tokens"], meta["decode_steps"]
    summaries = {s["run"]: s for s in summarize.summarize(prompt_dir, experiment)}
    baseline = summaries["cpu-baseline"]
    has_trace = exists(prompt_dir / "cpu-op-trace.jsonl")
    has_metal = exists(prompt_dir / "metal-baseline.jsonl")
    controls = datasets.controls_requested(prompt_dir)
    has_cuda_cpu = controls and exists(prompt_dir / "cuda-cpu-baseline.jsonl")
    has_cuda_cpu_no_host = controls and exists(prompt_dir / "cuda-cpu-no-host-baseline.jsonl")
    has_cuda = exists(prompt_dir / "cuda-baseline.jsonl")
    profile_checks, backend_checks, _ = [], [], None
    check = {}
    try:
        profile_checks, backend_checks, _ = validate.validate(prompt_dir, experiment, [n], skip_metal=not has_metal,
                                                              require_profile=has_trace)
        check["traced_vs_untraced_logits_bit_identical"] = all(c["bit_identical"] for c in profile_checks) if has_trace else None
    except ValueError as error:      # the instrumented run changed the logits: record it, do not hide it
        check["traced_vs_untraced_logits_bit_identical"] = False
        check["error"] = str(error)
    check["cpu_vs_metal"] = backend_checks
    check["cpu_vs_cuda_build_cpu"] = validate.compare_backend(
        prompt_dir, experiment, [n], "cuda-cpu") if has_cuda_cpu else []
    check["cpu_vs_cuda_build_no_host"] = validate.compare_backend(
        prompt_dir, experiment, [n], "cuda-cpu-no-host") if has_cuda_cpu_no_host else []
    check["cpu_vs_cuda"] = validate.compare_backend(prompt_dir, experiment, [n], "cuda") if has_cuda else []
    prefix = f"prompts/{prompt_id}/"
    trace_file = "cpu-op-trace.jsonl.gz" if (prompt_dir / "cpu-op-trace.jsonl.gz").is_file() else "cpu-op-trace.jsonl"
    row = {
        "prompt_id": prompt_id, "description": entry.get("description", ""),
        "prompt_file": f"core/prompts/{entry['file']}", "prompt_sha256": _sha256(prompt_set_dir / entry["file"]),
        "prompt_tokens": n,
        "continuation": {"mode": meta["mode"], "steps": steps, "source": "core/prompts/continuation.txt",
                         "source_sha256": _sha256(CONTINUATION)},
        "config": {"threads": meta["threads"], "flash_attention": meta["flash_attention"], "n_ubatch": meta["n_ubatch"],
                   "layers": meta["layers"], "untraced_repetitions": meta["reps"]},
        "timing": {"cpu": _timing(baseline, prompt_dir / "cpu-baseline.jsonl", n)},
        "trace": trace_block(ops.aggregate(prompt_dir, experiment)[0], steps, baseline) if has_trace else None,
        "checks": check,
        "graphs": _graphs(run_dir, prompt_id),
        "files": {"cpu_timings": prefix + "cpu-baseline.jsonl", "cpu_trace": prefix + trace_file if has_trace else None,
                  "tokens": prefix + "cpu-baseline-tokens.json"}}
    if has_metal:
        row["timing"]["metal"] = _timing(summaries["metal-baseline"], prompt_dir / "metal-baseline.jsonl", n)
    if has_cuda_cpu:
        row["timing"]["cuda_build_cpu"] = _timing(
            summaries["cuda-cpu-baseline"], prompt_dir / "cuda-cpu-baseline.jsonl", n)
    if has_cuda_cpu_no_host:
        row["timing"]["cuda_build_cpu_no_host"] = _timing(
            summaries["cuda-cpu-no-host-baseline"], prompt_dir / "cuda-cpu-no-host-baseline.jsonl", n)
    if has_cuda:
        row["timing"]["cuda"] = _timing(summaries["cuda-baseline"], prompt_dir / "cuda-baseline.jsonl", n)
    if has_trace:
        row["trace"]["repetitions_traced"] = row["trace"]["prefill"]["repetitions"]
    identities = datasets.read(prompt_dir)
    names = {"cpu": "cpu", "metal": "metal", "cuda_build_cpu": "cuda-cpu",
             "cuda_build_cpu_no_host": "cuda-cpu-no-host", "cuda": "cuda"}
    for key, timing in row["timing"].items():
        name = names[key]
        identity = next((item for item in identities if item["name"] == name), None)
        timing["dataset"] = identity
        timing["dataset_label"] = datasets.label(identity)
        timing["actual_backend"] = _metadata(prompt_dir / f"{name}-baseline.jsonl").get("backend")
    return row


def build(run_dir, experiment=None, prompt_set=None):
    """Returns {"prompts": [row, ...], ...} for every prompt folder found in the run, in prompt-set order."""
    run_dir = Path(run_dir)
    experiment = experiment or load_experiment(DEFAULT_EXPERIMENT)
    prompt_set = Path(prompt_set) if prompt_set else DEFAULT_PROMPT_SET
    listing = json.loads(prompt_set.read_text(encoding="utf-8"))
    rows = []
    for entry in listing["prompts"]:
        prompt_dir = run_dir / "prompts" / entry["id"]
        if exists(prompt_dir / "cpu-baseline.jsonl"):
            rows.append(build_row(run_dir, experiment, entry, prompt_dir, prompt_set.parent))
    return {"schema_version": 1, "datasets": datasets.read(run_dir),
            "note": "One row per prompt. Timings are host wall-clock medians for each execution dataset; trace shares are CPU operation intervals. "
                    "Neither is accelerator area, bandwidth or speedup, and results from different hosts are not comparable.",
            "prompts": rows}


def write(run_dir, experiment=None, prompt_set=None):
    result = build(run_dir, experiment, prompt_set)
    if result["prompts"]:
        write_json(Path(run_dir) / "prompt-profiles.json", result, trailing_newline=True)
    return result
