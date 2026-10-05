#!/usr/bin/env python3
"""SiliconBadgers profiling runner. Normally started through unix/run.sh or windows/run.ps1.

  profile   one command: build the pinned runtime, run baselines, traces and graph captures,
            then write summaries, validation, the figure and run-manifest.json to results/<run>/
  doctor    check tools, fetch the pinned source, test the trace patch and CMake configure (no model)
  analyze   re-run the analysis and rebuild the graph reports/SVGs for an existing run folder
            (captured and raw files are only read)
"""
import argparse
import json
from pathlib import Path
import sys

import common
import pipeline
import platform_profile
import runpaths
from common import Context, REPO
from policy import DEFAULT_EXPERIMENT, load_experiment


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform-dir", help="unix or windows (default: detected from the OS)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile", help="Run the whole pipeline into a new results/ folder")
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--work-dir", type=Path, default=REPO / "work", help="llama.cpp checkout, model and builds (not in Git)")
    p.add_argument("--results-root", type=Path, default=REPO / "results")
    p.add_argument("--lengths", type=positive, nargs="+")
    p.add_argument("--threads", type=positive)
    p.add_argument("--repetitions", type=positive)
    p.add_argument("--trace-repetitions", type=positive)
    p.add_argument("--decode-steps", type=positive, help="Continuation length: teacher-forced single-token steps after the prompt")
    p.add_argument("--long-continuation", action="store_true",
                   help="Use the experiment's long continuation (256 steps) unless --decode-steps is given")
    p.add_argument("--jobs", type=positive, default=6, help="Parallel build jobs")
    p.add_argument("--metal", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--skip-metal", action="store_true")
    p.add_argument("--cuda", choices=["on", "off"], default="off",
                   help="Also run uninstrumented timings on CUDA; CPU traces and graph capture stay canonical")
    p.add_argument("--no-trace", action="store_true", help="Skip the patched runtime and operation traces")
    p.add_argument("--no-graphs", action="store_true", help="Skip scheduled-graph capture")
    p.add_argument("--include-diagnostic", action="store_true",
                   help="Also trace the diagnostic cases (8K); their timing stays excluded from conclusions")
    p.add_argument("--no-prompt-graphs", action="store_true", help="Skip the one-graph-per-prompt captures")
    p.add_argument("--no-prompt-timings", action="store_true", help="Skip the per-prompt timings and traces")
    p.add_argument("--prompts-only", action="store_true",
                   help="Only the per-prompt stages: graphs, timings and traces (no fixed-length runs or figure)")
    p.add_argument("--prompt-set", type=Path, default=REPO / "core" / "prompts" / "prompts.json",
                   help="prompts.json listing the prompts to capture (default: the 8 in core/prompts)")
    p.add_argument("--smoke", action="store_true", help="Tiny run: 128 tokens, 1 repetition, 8 decode steps")
    p.add_argument("--model", type=Path, help="Reuse an existing model file after size/SHA256 verification")
    p.add_argument("--dry-run", action="store_true", help="Print every command; run nothing and write nothing")
    d = sub.add_parser("doctor", help="Preflight: tools, pinned source, trace patch, CMake configure (no model, no compile)")
    for action in p._actions[1:]:
        if action.dest not in ("help", "dry_run"):
            d._add_action(action)
    d.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("analyze", help="Re-run the analysis for an existing run folder")
    p.add_argument("run_dir", type=Path)
    return parser


def make_context(args):
    exp = load_experiment(args.experiment)
    work = exp["workload"]
    profile, os_label = platform_profile.load(args.platform_dir)
    lengths = args.lengths or work["prompt_lengths"]
    graph_lengths = exp["graphs"]["prompt_lengths"]
    if args.smoke:
        lengths, graph_lengths = lengths[:1] if not args.lengths else lengths, graph_lengths[:1]
    reps = args.repetitions or (1 if args.smoke else work["repetitions"])
    steps = args.decode_steps or (8 if args.smoke else work["long_continuation_steps"] if args.long_continuation
                                  else work["decode_steps"])
    if not args.prompts_only:
        too_long = [n for n in lengths if n + steps > work["prompt_source_tokens"]]
        if too_long:
            sys.exit(f"error: prompt length(s) {too_long} plus {steps} continuation steps exceed the "
                     f"{work['prompt_source_tokens']}-token source passage. Shorten --decode-steps to at most "
                     f"{work['prompt_source_tokens'] - max(too_long)} or drop those lengths.")
    ctx = Context(
        experiment=exp, platform=profile, os_label=os_label, work=args.work_dir.resolve(),
        run_dir=runpaths.create_run_dir(args.results_root, os_label, create=False),
        lengths=lengths, threads=args.threads or work["threads"], repetitions=reps,
        trace_repetitions=args.trace_repetitions or (1 if args.smoke else work["trace_repetitions"]),
        decode_steps=steps,
        graph_lengths=graph_lengths, jobs=args.jobs, metal=args.metal, skip_metal=args.skip_metal, cuda=args.cuda,
        run_trace=not args.no_trace, run_graphs=not args.no_graphs, include_diagnostic=args.include_diagnostic,
        model_override=args.model, dry_run=args.dry_run, smoke=args.smoke,
        run_prompt_graphs=args.prompts_only or not (args.no_prompt_graphs or args.smoke),
        prompts_only=args.prompts_only, prompt_set=args.prompt_set.resolve(),
        run_prompt_timings=args.prompts_only or not (args.no_prompt_timings or args.smoke))
    runpaths.assert_writable(ctx.work)
    return ctx


def profile_command(args, argv):
    ctx = make_context(args)
    print(f"Run folder: {ctx.run_dir}" + ("  (dry run: nothing is created)" if ctx.dry_run else ""))
    pipeline.run_profile(ctx, argv)


def doctor_command(args):
    ctx = make_context(args)
    ctx.state["metal"] = platform_profile.metal_available(ctx.platform, ctx.os_label) and args.metal != "off"
    ctx.state["cuda"] = args.cuda == "on"
    if not pipeline.doctor(ctx):
        sys.exit(1)


def analyze_command(args):
    run_dir = args.run_dir.resolve()
    runpaths.assert_writable(run_dir)
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    exp = load_experiment(manifest["experiment"]["name"])
    cfg = manifest["config"]
    result = pipeline.analyze_run(run_dir, exp, cfg["prompt_lengths"], cfg["trace"], cfg["backends"], manifest["plot_subtitle"])
    manifest["analysis"] = result
    (run_dir / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Analysis written to", run_dir / "analysis")
    graphs, drawn = pipeline.reanalyze_graphs(run_dir)
    rows = pipeline.rebuild_prompt_profiles(run_dir, exp, manifest)
    if rows:
        print("Per-prompt profiles rebuilt:", run_dir / "prompt-profiles.json", f"({rows} prompts)")
    for name in graphs:
        print("Graph report and index rebuilt:", run_dir / "graphs" / name / "index.html")
    if graphs and not drawn:
        print("Note: Graphviz 'dot' not found, so layer SVGs were not drawn (DOT files are written). "
              "Install Graphviz and run analyze again.")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    if args.command == "profile":
        profile_command(args, ["cli.py"] + argv)
    elif args.command == "doctor":
        doctor_command(args)
    else:
        analyze_command(args)


if __name__ == "__main__":
    main()
