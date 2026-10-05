"""Per-prompt timings and operation traces: one untraced run and one traced run per prompt.

Each prompt gets its own folder, results/<run>/prompts/<id>/, using the same file names as the fixed-length
runs (cpu-baseline*, cpu-profile*, cpu-op-trace.jsonl) so the existing analysis code reads them unchanged.
"""
import os

from build import build_dir
from capture import load_prompt_set
from common import REPO, exe_path
import measure

CONTINUATION = REPO / "core" / "prompts" / "continuation.txt"


def prompt_dir(ctx, prompt_id):
    return ctx.run_dir / "prompts" / prompt_id


def _run(ctx, exe, model, ngl, reps, prompt_file, prefix, trace=None):
    command = [exe, model, ngl, ctx.threads, reps, ctx.decode_steps, prompt_file, CONTINUATION, prefix]
    if ctx.dry_run:
        ctx.run(command)
        return
    env = dict(os.environ)
    env.pop("SB_CPU_TRACE", None)
    env.pop("SB_CPU_PHASE", None)
    if trace:
        env["SB_CPU_TRACE"] = str(trace)
    with prefix.with_suffix(".jsonl").open("x") as stdout, prefix.with_suffix(".log").open("x") as stderr:
        ctx.run(command, env=env, stdout=stdout, stderr=stderr)


def time_prompts(ctx, model):
    """Untraced timings for requested backends, then a traced CPU run, for every prompt."""
    traced = exe_path(ctx, build_dir(ctx, "profile"), "sb-profile-prompt")
    done = []
    for prompt_id, file, _ in load_prompt_set(ctx.prompt_set):
        out = prompt_dir(ctx, prompt_id)
        if not ctx.dry_run:
            out.mkdir(parents=True, exist_ok=False)
        for backend, variant, ngl in measure.backends(ctx):
            plain = exe_path(ctx, build_dir(ctx, variant), "sb-profile-prompt")
            _run(ctx, plain, model, ngl, ctx.repetitions, file, out / f"{backend}-baseline")
        if ctx.run_trace:
            _run(ctx, traced, model, 0, ctx.trace_repetitions, file, out / "cpu-profile",
                 trace=out / "cpu-op-trace.jsonl")
        done.append(prompt_id)
    return done
