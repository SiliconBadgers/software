"""Timing baselines and CPU operation traces, using the recorded 2026-09-22 file naming."""
import json
import os

from build import build_dir
from common import exe_path


def _measure(ctx, exe, model, prefix, lengths, ngl, repetitions, no_host=False, trace=None):
    """Run the harness once: sb-profile MODEL NGL THREADS REPS STEPS LENGTHS OUTPUT_PREFIX NO_HOST."""
    out_dir = ctx.run_dir
    target = out_dir / prefix
    command = [exe, model, ngl, ctx.threads, repetitions, ctx.decode_steps, ",".join(map(str, lengths)), target,
               int(no_host)]
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
    return read_backend_metadata(out_dir / (prefix + ".jsonl"))


def _cases(ctx):
    """[(tag, lengths)] following the 2026-09-22 convention: separate cases such as 8K get their own file."""
    separate = ctx.experiment["policy"]["separate_case_prefixes"]
    regular = [n for n in ctx.lengths if str(n) not in separate]
    cases = [("baseline", regular)] if regular else []
    cases += [(separate[str(n)], [n]) for n in ctx.lengths if str(n) in separate]
    return cases


def backends(ctx):
    """Timing plans as (result name, build variant, GPU layers, no_host, dataset role)."""
    metal = bool(ctx.state.get("metal")) and not ctx.skip_metal
    cuda_mode = ctx.state.get("cuda_mode", "off")
    if cuda_mode == "only":
        return [("cuda", "cuda", 99, False, "accelerated_execution")]
    plans = [("cpu", "baseline", 0, False, "cpu_reference")]
    plans += ([("metal", "baseline", 99, False, "accelerated_execution")] if metal else [])
    if cuda_mode == "on":
        if getattr(ctx, "cuda_controls", False) or getattr(ctx, "cuda_no_host_control", False):
            plans.append(("cuda-cpu", "cuda", 0, False, "cuda_build_cpu_control"))
        if getattr(ctx, "cuda_no_host_control", False):
            plans.append(("cuda-cpu-no-host", "cuda", 0, True, "cuda_build_cpu_no_host_control"))
        plans.append(("cuda", "cuda", 99, False, "accelerated_execution"))
    return plans


def dataset_specs(ctx):
    return [{"name": name, "role": role, "build": variant, "requested_ngl": ngl, "no_host": no_host}
            for name, variant, ngl, no_host, role in backends(ctx)]


def read_backend_metadata(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "metadata":
                actual = row.get("backend")
                if actual is None:
                    raise ValueError(f"Harness did not report actual backend placement: {path}")
                return actual
    raise ValueError(f"Harness did not report metadata: {path}")


def check_fallback(name, actual):
    if actual.get("cpu_fallback"):
        raise ValueError(f"Accelerator request fell back to CPU: {name}")


def baseline(ctx, model):
    datasets = []
    ctx.state["datasets"] = datasets
    for backend, variant, ngl, no_host, role in backends(ctx):
        exe = exe_path(ctx, build_dir(ctx, variant), "sb-profile")
        row = {"name": backend, "role": role, "build": variant, "requested_ngl": ngl, "no_host": no_host,
               "cases": []}
        datasets.append(row)
        for tag, lengths in _cases(ctx):
            measured = _measure(ctx, exe, model, f"{backend}-{tag}", lengths, ngl, ctx.repetitions, no_host=no_host)
            if measured is not None:
                row["cases"].append({"file": f"{backend}-{tag}.jsonl", "actual": measured})
                row["cpu_fallback"] = any(case["actual"].get("cpu_fallback") for case in row["cases"])
                check_fallback(backend, measured)
    return datasets


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
