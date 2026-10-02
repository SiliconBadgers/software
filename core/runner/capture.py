"""Scheduled-graph capture (prefill + one decode step): the fixed-length graphs, and one graph per prompt."""
import hashlib
import json
from pathlib import Path

from build import build_dir
from common import REPO, exe_path


def graph_dirs(ctx):
    graphs = ctx.experiment["graphs"]
    return [(n, fa, ctx.run_dir / "graphs" / f"pp{n}-fa-{fa}") for n in ctx.graph_lengths
            for fa in graphs["flash_attention"]]


def capture(ctx, model):
    import graph as graph_analysis
    exe = exe_path(ctx, build_dir(ctx, "baseline"), "sb-capture-graph")
    multiple = ctx.experiment["graphs"]["context_multiple"]
    captured = []
    for n, fa, out in graph_dirs(ctx):
        ctx.run([exe, model, out, n, n * multiple, fa, ctx.threads])
        if not ctx.dry_run:
            graph_analysis.analyze_directory(out)
            captured.append({"directory": str(out.relative_to(ctx.run_dir)).replace("\\", "/"),
                             "prompt_tokens": n, "context_capacity": n * multiple, "flash_attention": fa})
    return captured


def load_prompt_set(path):
    """Returns [(id, file path, description)] from a prompts.json; every file must exist and be non-empty."""
    path = Path(path)
    listing = json.loads(path.read_text(encoding="utf-8"))
    prompts = []
    for entry in listing["prompts"]:
        file = path.parent / entry["file"]
        if not file.is_file() or file.stat().st_size == 0:
            raise FileNotFoundError(f"Prompt file missing or empty: {file}")
        prompts.append((entry["id"], file, entry.get("description", "")))
    return prompts


def capture_prompts(ctx, model):
    """One graph per prompt and flash-attention setting, using each prompt's natural token count
    (prompt_tokens=0, context=0 tell sb-capture-graph to size both from the file)."""
    import graph as graph_analysis
    exe = exe_path(ctx, build_dir(ctx, "baseline"), "sb-capture-graph")
    captured = []
    for prompt_id, file, description in load_prompt_set(ctx.prompt_set):
        for fa in ctx.experiment["graphs"]["flash_attention"]:
            out = ctx.run_dir / "graphs" / f"prompt-{prompt_id}-fa-{fa}"
            ctx.run([exe, model, out, 0, 0, fa, ctx.threads, file])
            if ctx.dry_run:
                continue
            graph_analysis.analyze_directory(out)
            meta = json.loads((out / "prefill.summary.json").read_text(encoding="utf-8"))["metadata"]
            captured.append({
                "directory": str(out.relative_to(ctx.run_dir)).replace("\\", "/"), "prompt_id": prompt_id,
                "prompt_file": str(file.relative_to(REPO)).replace("\\", "/") if REPO in file.parents else str(file),
                "prompt_sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "description": description,
                "prompt_tokens": meta["prompt_tokens"], "context_capacity": meta["context_capacity"],
                "flash_attention": fa})
    if not ctx.dry_run:
        graph_analysis.compare_prompt_graphs(ctx.run_dir)
    return captured
