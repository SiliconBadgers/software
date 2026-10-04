"""Measured CPU operation timings (cpu-op-trace.jsonl.gz) matched to captured graph nodes.

The instrumented run records `elapsed_us` per scheduled node and repetition. Only the fa-on graphs match the
profiled configuration, so only they have a trace. Times are averaged over repetitions; for decode only the
first step (`decode0`) is used because that is the step the captured decode graph describes.
"""
import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

_PHASE = re.compile(r"^p(\d+)_r(\d+)_(prefill|decode0)$")


def trace_path(run_dir, workload_name):
    run_dir = Path(run_dir)
    if re.match(r"^pp\d+-fa-on$", workload_name):
        path = run_dir / "cpu-op-trace.jsonl.gz"
    else:
        m = re.match(r"^prompt-(.+)-fa-on$", workload_name)
        if not m:
            return None
        path = run_dir / "prompts" / m.group(1) / "cpu-op-trace.jsonl.gz"
    return path if path.is_file() else None


@lru_cache(maxsize=8)
def _load(path):
    """{(tokens, phase): {scheduled position: mean seconds}}. A phase that ran as several graphs in one repetition
    (a prompt longer than the 512-token micro-batch is split) does not correspond to the single captured graph, so
    it is dropped rather than averaged across the pieces."""
    acc, graph_ids = {}, {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            x = json.loads(line)
            m = _PHASE.match(x["phase"])
            if not m or x["graph_status"] != 0 or x["elapsed_us"] < 0:
                continue
            key = (int(m.group(1)), "prefill" if m.group(3) == "prefill" else "decode")
            acc.setdefault(key, {}).setdefault(x["node_index"], []).append(x["elapsed_us"] * 1e-6)
            graph_ids.setdefault((key, int(m.group(2))), set()).add(x["graph_id"])
    split = {key for (key, _rep), ids in graph_ids.items() if len(ids) > 1}
    return {key: {node: sum(v) / len(v) for node, v in nodes.items()}
            for key, nodes in acc.items() if key not in split}


def load_trace(run_dir, workload):
    """{'prefill': {scheduled position: seconds}, 'decode': {...}} for an fa-on workload, else None."""
    path = trace_path(run_dir, workload.name)
    if path is None:
        return None
    data = _load(str(path))
    tokens = workload.meta["prompt_tokens"]
    out = {phase: data.get((tokens, phase)) for phase in ("prefill", "decode")}
    return out if any(out.values()) else None
