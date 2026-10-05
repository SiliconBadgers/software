"""The single `profile` command: ensure -> baseline -> trace -> graphs -> compress -> analyze -> manifest."""
from datetime import datetime, timezone
import json
import shlex
import sys
import traceback

import build
import capture
import fetch
import hostinfo
import measure
import platform_profile
import prompt_timing
import runpaths
from common import gzip_in_place


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _subtitle(host, experiment, threads, backends):
    parts = [host["cpu_model"], experiment["model"]["label"], f"{threads} CPU threads"]
    accelerated = [name.upper() if name == "cuda" else name.title()
                   for name in backends if name not in ("cpu", "cuda-cpu")]
    if accelerated:
        parts.append(" + ".join(accelerated) + " measured")
    parts.append("one sequence")
    return " · ".join(parts)


def _write_manifest(ctx, manifest):
    if not ctx.dry_run:
        (ctx.run_dir / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def compress(ctx):
    """Logits and op traces are large and repetitive; readers accept .gz transparently."""
    if ctx.dry_run:
        return
    for pattern in ("*.f32", "*op-trace.jsonl"):
        for path in sorted(ctx.run_dir.rglob(pattern)):
            gzip_in_place(path)


def analyze_run(run_dir, experiment, lengths, has_trace, backends, subtitle):
    """Write summaries, validation and the figure into run_dir/analysis. Raw files are only read."""
    import decode_curve
    import ops
    import plot
    import summarize
    import validate
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    summarize.write_outputs(out, summarize.summarize(run_dir, experiment))
    curves = decode_curve.summarize_run(run_dir, experiment)
    decode_curve.write_outputs(out, curves)
    decode_curve.plot(out, curves)
    if has_trace:
        ops.write_outputs(out, *ops.aggregate(run_dir, experiment))
    has_cpu = "cpu" in backends
    if has_cpu:
        profile_checks, backend_checks, missing = validate.validate(
            run_dir, experiment, lengths, skip_metal="metal" not in backends, require_profile=has_trace)
    else:
        profile_checks, backend_checks, missing = [], [], []
    validate.write_outputs(out, profile_checks, backend_checks)
    cuda_checks = validate.compare_backend(run_dir, experiment, lengths, "cuda") if has_cpu and "cuda" in backends else []
    if cuda_checks:
        validate.write_backend_output(out, "cuda", cuda_checks)
    plot.draw(out, out, experiment, subtitle)
    return {"decode_curves": len(curves), "profile_checks": len(profile_checks),
            "profile_bit_identical": all(c["bit_identical"] for c in profile_checks) if has_trace else None,
            "backend_checks": len(backend_checks) + len(cuda_checks),
            "backend_checks_by_backend": {"metal": len(backend_checks), "cuda": len(cuda_checks)},
            "cuda_reference": "canonical cpu" if cuda_checks else None,
            "missing_optional_diagnostic_profile": missing}


def reanalyze_graphs(run_dir):
    """Re-derive every graphs/<name>/ folder (reports, CSVs, DOT, index.html, and layer SVGs if Graphviz is
    installed) from its captured prefill.json/decode.json, which are only read. Returns (names, svgs_drawn)."""
    import graph
    names, drawn = [], True
    for directory in sorted(p for p in (run_dir / "graphs").glob("*") if (p / "prefill.json").is_file()
                            and (p / "decode.json").is_file()):
        _, have_dot = graph.analyze_directory(directory)
        drawn &= have_dot
        names.append(directory.name)
    graph.compare_prompt_graphs(run_dir)      # no-op for runs without per-prompt graphs
    return names, drawn


def rebuild_prompt_profiles(run_dir, experiment, manifest):
    """Rebuild prompt-profiles.json from the per-prompt folders of a run; returns the number of prompt rows."""
    import prompts
    settings = manifest.get("config", {}).get("prompt_timings") or {}
    return len(prompts.write(run_dir, experiment, settings.get("prompt_set"))["prompts"])


def doctor(ctx):
    """Preflight without the model download or a compile: tools, pinned source, patch, CMake configure."""
    import shutil
    import subprocess
    from common import tool
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))

    print(f"OS label: {ctx.os_label}   platform profile: {ctx.platform['directory']}")
    try:
        compilers = build.find_compilers(ctx)
        report("C++ compiler", compilers["cxx"] is not None or not ctx.platform["compilers"]["cxx"],
               compilers["cxx"] or "(none listed; CMake will choose, honouring CC/CXX)")
    except RuntimeError as error:
        report("C++ compiler", False, str(error))
        compilers = {"c": None, "cxx": None}
    for name in ("cmake", "ninja", "git"):
        try:
            report(name, True, tool(name) if name != "git" else shutil.which("git"))
        except RuntimeError as error:
            report(name, False, str(error))
    print("INFO  Graphviz dot:", shutil.which("dot") or "not found (graph DOT files are still written; SVG diagrams are skipped)")
    if not ok:
        print()
        print("Fix the failures above, then re-run doctor.")
        return False
    fetch.ensure_source(ctx)
    lf_patch = build.write_lf_patch(ctx)
    try:
        ctx.run(["git", "-C", ctx.work / "llama.cpp", "apply", "--check", lf_patch])
        report("trace patch applies to the pinned source", True)
    except subprocess.CalledProcessError:
        report("trace patch applies to the pinned source", False)
    cuda_mode = ctx.state.get("cuda_mode", "off")
    variants = []
    if cuda_mode != "only":
        variants.append(("baseline", build.BASELINE_TARGETS, "canonical CPU"))
    if cuda_mode != "off":
        variants.append(("cuda", build.CUDA_TARGETS, "CUDA"))
    for variant, targets, label in variants:
        try:
            build.build_variant(ctx, variant, targets, compile=False)
            report(f"CMake configures the {label} harness with the pinned llama.cpp", True)
        except (subprocess.CalledProcessError, RuntimeError, TypeError) as error:
            report(f"CMake configures the {label} harness with the pinned llama.cpp", False, str(error))
    print()
    print("Doctor: ready for `profile`." if ok else "Doctor: problems found.")
    return ok


def run_profile(ctx, argv):
    exp = ctx.experiment
    metal_ok = platform_profile.metal_available(ctx.platform, ctx.os_label)
    if ctx.metal == "on" and not metal_ok:
        raise ValueError("Metal is only offered on macOS/arm64")
    cuda_compiler = platform_profile.cuda_compiler()
    if ctx.cuda != "off" and not cuda_compiler:
        raise ValueError("CUDA requested but nvcc was not found; install the CUDA Toolkit or use --cuda off")
    ctx.state["metal"] = metal_ok if ctx.metal == "auto" else ctx.metal == "on"
    ctx.state["cuda_mode"] = ctx.cuda
    backends = [name for name, _, _ in measure.backends(ctx)]
    compilers = build.find_compilers(ctx)
    host = hostinfo.collect(ctx.os_label, compilers, cuda_compiler if ctx.cuda != "off" else None)
    manifest = {
        "schema_version": 1,
        "run": {"name": ctx.run_dir.name, "started_at": _now(), "finished_at": None, "status": "running",
                "command": " ".join(shlex.quote(a) for a in argv), "dry_run": ctx.dry_run},
        "host": host,
        "plot_subtitle": _subtitle(host, exp, ctx.threads, backends),
        "experiment": {"name": exp["name"], "llama_cpp": exp["llama_cpp"], "model": exp["model"],
                       "policy": exp["policy"]},
        "config": {"prompt_lengths": ctx.lengths, "threads": ctx.threads, "repetitions": ctx.repetitions,
                   "trace_repetitions": ctx.trace_repetitions, "decode_steps": ctx.decode_steps,
                   "backends": backends, "trace": ctx.run_trace,
                   "include_diagnostic_cases": ctx.include_diagnostic,
                   "graphs": ctx.run_graphs and {"prompt_lengths": ctx.graph_lengths,
                                                  "flash_attention": exp["graphs"]["flash_attention"]}},
        "comparability": "Results from different hosts are not comparable to each other or to the recorded "
                         "2026-09-22 Apple M5 Pro baseline. CPU operation shares do not describe accelerator "
                         "area, bandwidth or speedup.",
        "graphs": [], "analysis": None, "build": None, "cuda_build": None, "profile_build": None,
    }
    manifest["config"]["smoke"] = ctx.smoke
    manifest["config"]["prompts_only"] = ctx.prompts_only
    manifest["config"]["prompt_graphs"] = ctx.run_prompt_graphs and {
        "prompt_set": str(ctx.prompt_set), "flash_attention": exp["graphs"]["flash_attention"]}
    manifest["prompt_graphs"] = []
    manifest["config"]["prompt_timings"] = ctx.run_prompt_timings and {
        "prompt_set": str(ctx.prompt_set), "continuation": "core/prompts/continuation.txt",
        "repetitions": ctx.repetitions, "trace_repetitions": ctx.trace_repetitions if ctx.run_trace else 0,
        "decode_steps": ctx.decode_steps}
    manifest["prompt_profiles"] = None
    if not ctx.dry_run:
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(ctx, manifest)
    try:
        ctx.work.mkdir(parents=True, exist_ok=True) if not ctx.dry_run else None
        print(f"\n== ensure: pinned llama.cpp {exp['llama_cpp']['commit'][:12]} and model {exp['model']['file']}")
        fetch.ensure_source(ctx)
        model = fetch.ensure_model(ctx)
        if ctx.cuda != "only":
            print("\n== build: pristine canonical CPU runtime")
            build.ensure_baseline(ctx)
        if ctx.cuda != "off":
            print("\n== build: separate pristine CUDA runtime")
            build.ensure_cuda(ctx)
        if not ctx.prompts_only:
            print("\n== baseline: uninstrumented timings and saved logits")
            measure.baseline(ctx, model)
            if ctx.run_trace:
                print("\n== trace: patched runtime, CPU operation traces")
                if ctx.decode_steps > 64:
                    print(f"Note: a trace records every one of the {ctx.decode_steps} decode steps, roughly 0.4 MB of "
                          "text per step and repetition before compression; expect large temporary files and a "
                          "slower traced run.")
                build.ensure_profile(ctx)
                measure.trace(ctx, model)
            if ctx.run_graphs:
                print("\n== graphs: scheduled dataflow graphs (prefill + one decode step)")
                manifest["graphs"] = capture.capture(ctx, model)
        if ctx.run_prompt_graphs:
            print("\n== prompt graphs: one scheduled graph per prompt and flash-attention setting")
            manifest["prompt_graphs"] = capture.capture_prompts(ctx, model)
        if ctx.run_prompt_timings:
            print("\n== prompt timings: per-prompt untraced timings and operation traces")
            if ctx.run_trace:
                build.ensure_profile(ctx)
            manifest["prompt_timings"] = prompt_timing.time_prompts(ctx, model)
        manifest["build"] = build.build_config(ctx, "baseline") if ctx.cuda != "only" else None
        manifest["cuda_build"] = build.build_config(ctx, "cuda") if ctx.cuda != "off" else None
        manifest["profile_build"] = build.build_config(ctx, "profile") if ctx.run_trace else None
        compress(ctx)
        if ctx.run_prompt_timings and not ctx.dry_run:
            manifest["prompt_profiles"] = {"file": "prompt-profiles.json",
                                           "prompts": rebuild_prompt_profiles(ctx.run_dir, exp, manifest)}
        if not ctx.prompts_only:
            print("\n== analyze")
            if not ctx.dry_run:
                manifest["analysis"] = analyze_run(ctx.run_dir, exp, ctx.lengths, ctx.run_trace, backends,
                                                   manifest["plot_subtitle"])
        manifest["run"]["status"] = "dry-run" if ctx.dry_run else "complete"
    except BaseException:
        manifest["run"]["status"] = "failed"
        manifest["run"]["error"] = traceback.format_exc(limit=3)
        raise
    finally:
        manifest["run"]["finished_at"] = _now()
        _write_manifest(ctx, manifest)
    print("\nRun folder:", ctx.run_dir)
    return ctx.run_dir
