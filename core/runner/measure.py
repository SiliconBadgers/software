"""Timing baselines and CPU operation traces, using the recorded 2026-09-22 file naming."""
import json
import os

from build import build_dir
from common import exe_path


def _measure(ctx, exe, model, prefix, lengths, ngl, repetitions, trace=None):
    """Run the harness once: sb-profile MODEL NGL THREADS REPS STEPS LENGTHS OUTPUT_PREFIX."""
    out_dir = ctx.run_dir
    target = out_dir / prefix
    command = [exe, model, ngl, ctx.threads, repetitions, ctx.decode_steps, ",".join(map(str, lengths)), target]
    if ctx.dry_run:
        ctx.run(command)
        return
    if list(out_dir.glob(prefix + "*")):
        raise ValueError(f"Output prefix already exists: {target}")
    if trace and trace.exists():
        raise ValueError(f"Trace logger appends; refusing existing trace: {trace}")
    env = dict(os.environ)
    env.pop("SB_CPU_TRACE", None)
    env.pop("SB_CPU_PHASE", None)
    if trace:
        env["SB_CPU_TRACE"] = str(trace)
    with (out_dir / (prefix + ".jsonl")).open("x") as stdout, (out_dir / (prefix + ".log")).open("x") as stderr:
        ctx.run(command, env=env, stdout=stdout, stderr=stderr)


def _cases(ctx):
    """[(tag, lengths)] following the 2026-09-22 convention: separate cases such as 8K get their own file."""
    separate = ctx.experiment["policy"]["separate_case_prefixes"]
    regular = [n for n in ctx.lengths if str(n) not in separate]
    cases = [("baseline", regular)] if regular else []
    cases += [(separate[str(n)], [n]) for n in ctx.lengths if str(n) in separate]
    return cases


def backends(ctx):
    metal = bool(ctx.state.get("metal")) and not ctx.skip_metal
    return [("cpu", 0)] + ([("metal", 99)] if metal else [])


def baseline(ctx, model):
    exe = exe_path(ctx, build_dir(ctx, "baseline"), "sb-profile")
    for backend, ngl in backends(ctx):
        for tag, lengths in _cases(ctx):
            _measure(ctx, exe, model, f"{backend}-{tag}", lengths, ngl, ctx.repetitions)


def trace(ctx, model):
    exe = exe_path(ctx, build_dir(ctx, "profile"), "sb-profile")
    if not ctx.dry_run:
        baseline_path = ctx.run_dir / "cpu-baseline.jsonl"
        if not baseline_path.is_file():
            raise ValueError("Collect CPU baselines in this run before enabling instrumentation")
        rows = [json.loads(l) for l in baseline_path.read_text().splitlines() if l.strip()]
        meta = next(r for r in rows if r["kind"] == "metadata")
        if meta["threads"] != ctx.threads or meta["decode_steps"] != ctx.decode_steps:
            raise ValueError("Trace threads/decode steps must match the baseline")
    for tag, lengths in _cases(ctx):
        if tag == "baseline":
            _measure(ctx, exe, model, "cpu-profile", lengths, 0, ctx.trace_repetitions,
                     trace=ctx.run_dir / "cpu-op-trace.jsonl")
        elif ctx.include_diagnostic:
            _measure(ctx, exe, model, f"cpu-{tag}-profile", lengths, 0, ctx.trace_repetitions,
                     trace=ctx.run_dir / f"cpu-{tag}-op-trace.jsonl")
