#!/usr/bin/env python3
"""Build the pinned runtime, then collect fresh baseline and optional CPU traces."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
COMMIT = MANIFEST["llama_cpp_commit"]
MODEL = MANIFEST["model"]
PATCH = ROOT / "sb-cpu-profile.patch"
RECORDED = ROOT / "results"


def outside_recorded(path):
    path = path.expanduser().resolve()
    if path == RECORDED or RECORDED in path.parents:
        raise ValueError("Recorded results are immutable; choose a separate directory")
    return path


def run(command, **kwargs):
    command = [str(arg) for arg in command]
    print("+", shlex.join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def tool(name):
    local = Path(sys.executable).parent / name
    found = str(local) if local.is_file() else shutil.which(name)
    if not found:
        raise RuntimeError(f"Missing {name}; install requirements.txt and a C/C++ toolchain")
    return found


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_model(path):
    if path.stat().st_size != MODEL["size_bytes"] or digest(path) != MODEL["sha256"]:
        raise ValueError(f"Model size/SHA256 mismatch: {path}")


def state_path(work):
    return work / "reproduction-state.json"


def load_state(work):
    state = json.loads(state_path(work).read_text())
    source = work / "llama.cpp"
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise ValueError("Runtime checkout no longer matches the pinned commit")
    verify_model(Path(state["model_path"]))
    return state


def save_state(work, state):
    state_path(work).write_text(json.dumps(state, indent=2) + "\n")


def build(work, state, variant, jobs):
    source, directory = work / "llama.cpp", work / ("build-" + variant)
    cmake = tool("cmake")
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    run([cmake, "-S", source, "-B", directory, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_EXAMPLES=ON",
         "-DLLAMA_OPENSSL=OFF", "-DGGML_NATIVE=ON", "-DGGML_METAL=" + ("ON" if state["metal"] else "OFF")], env=env)
    run([cmake, "--build", directory, "--target", "llama-bench", "llama-simple", "-j", str(jobs)], env=env)
    compiler = shlex.split(os.environ.get("CXX", "clang++" if sys.platform == "darwin" else "c++"))
    executable = work / ("profile-" + variant)
    run(compiler + ["-O3", "-std=c++17", ROOT / "profile.cpp", "-I", source / "include",
         "-I", source / "ggml/include", "-L", directory / "bin",
         "-Wl,-rpath," + str(directory / "bin"), "-lllama", "-lggml", "-lggml-base", "-lggml-cpu",
         "-o", executable])
    return executable


def setup(args, work):
    if work.exists() and any(work.iterdir()):
        raise ValueError("Setup requires a fresh, empty --work-dir; existing builds are never reset")
    work.mkdir(parents=True, exist_ok=True)
    source = work / "llama.cpp"
    run(["git", "init", source])
    run(["git", "-C", source, "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp.git"])
    run(["git", "-C", source, "fetch", "--depth=1", "origin", COMMIT])
    run(["git", "-C", source, "checkout", "--detach", COMMIT])
    model_path = args.model.expanduser().resolve() if args.model else work / "models" / MODEL["file"]
    if args.model is None:
        model_path.parent.mkdir(parents=True)
        url = f'https://huggingface.co/{MODEL["repo"]}/resolve/{MODEL["revision"]}/{MODEL["file"]}'
        partial = model_path.with_suffix(model_path.suffix + ".partial")
        print("Downloading pinned model:", url, flush=True)
        with urllib.request.urlopen(url) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        verify_model(partial)
        partial.rename(model_path)
    else:
        verify_model(model_path)
    metal_supported = sys.platform == "darwin" and platform.machine() == "arm64"
    if args.metal == "on" and not metal_supported:
        raise ValueError("This runner enables Metal only on Apple Silicon macOS")
    state = dict(commit=COMMIT, model_path=str(model_path), model_sha256=MODEL["sha256"],
                 metal=metal_supported if args.metal == "auto" else args.metal == "on",
                 host=platform.platform(), setup_at=datetime.now(timezone.utc).isoformat(),
                 baseline_ready=False, profile_ready=False)
    save_state(work, state)
    build(work, state, "baseline", args.jobs)
    state["baseline_ready"] = True
    save_state(work, state)


def measurement(work, state, output, prefix, lengths, ngl, args, *, trace=None):
    executable = work / ("profile-profile" if trace else "profile-baseline")
    target = output / prefix
    if list(output.glob(prefix + "*")):
        raise ValueError(f"Output prefix already exists: {target}")
    if trace and trace.exists():
        raise ValueError(f"Trace logger appends; refusing existing trace: {trace}")
    env = dict(os.environ)
    env.pop("SB_CPU_TRACE", None)
    env.pop("SB_CPU_PHASE", None)
    if trace:
        env["SB_CPU_TRACE"] = str(trace)
    with target.with_suffix(".jsonl").open("x") as stdout, target.with_suffix(".log").open("x") as stderr:
        run([executable, state["model_path"], str(ngl), str(args.threads), str(args.repetitions),
             str(args.decode_steps), ",".join(map(str, lengths)), target],
            env=env, stdout=stdout, stderr=stderr)


def baseline(args, work):
    state = load_state(work)
    if not state["baseline_ready"]:
        raise ValueError("Run setup successfully first")
    output = outside_recorded(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if any(n <= 0 for n in args.lengths) or args.decode_steps <= 0 or args.repetitions <= 0:
        raise ValueError("Lengths, decode steps and repetitions must be positive")
    for backend, ngl in [("cpu", 0)] + ([("metal", 99)] if state["metal"] and not args.skip_metal else []):
        short = [n for n in args.lengths if n != 8192]
        if short:
            measurement(work, state, output, backend + "-baseline", short, ngl, args)
        if 8192 in args.lengths:
            measurement(work, state, output, backend + "-8k", [8192], ngl, args)
    (output / "run-config.json").write_text(json.dumps(dict(state, lengths=args.lengths,
        threads=args.threads, repetitions=args.repetitions, decode_steps=args.decode_steps), indent=2) + "\n")
    print("Fresh baseline evidence:", output)


def trace(args, work):
    state = load_state(work)
    output = outside_recorded(args.results_dir)
    baseline_path = output / "cpu-baseline.jsonl"
    if not baseline_path.is_file():
        raise ValueError("Collect CPU baselines before enabling instrumentation")
    records = [json.loads(line) for line in baseline_path.read_text().splitlines() if line.strip()]
    meta = next(row for row in records if row["kind"] == "metadata")
    lengths = sorted({row["prompt_tokens"] for row in records if row["kind"] == "measurement"})
    if meta["threads"] != args.threads or meta["decode_steps"] != args.decode_steps:
        raise ValueError("Trace threads/decode steps must match this baseline")
    if list(output.glob("cpu-profile*")) or (output / "cpu-op-trace.jsonl").exists():
        raise ValueError("This result directory already contains a CPU trace run")
    if args.include_8k and not (output / "cpu-8k.jsonl").is_file():
        raise ValueError("--include-8k requires an 8K baseline in the same result directory")
    if not state["profile_ready"]:
        source = work / "llama.cpp"
        if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
            raise ValueError("Refusing to patch a modified runtime checkout; use a fresh workspace")
        run(["git", "-C", source, "apply", "--check", PATCH])
        run(["git", "-C", source, "apply", PATCH])
        build(work, state, "profile", args.jobs)
        state["profile_ready"] = True
        save_state(work, state)
    measurement(work, state, output, "cpu-profile", lengths, 0, args, trace=output / "cpu-op-trace.jsonl")
    if args.include_8k:
        measurement(work, state, output, "cpu-8k-profile", [8192], 0, args,
                    trace=output / "cpu-8k-op-trace.jsonl")
    print("CPU traces:", output, "(8K timing remains diagnostic)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("setup", help="Download/verify model and build pristine baseline runtime")
    p.add_argument("--model", type=Path, help="Reuse an existing model after size/SHA256 verification")
    p.add_argument("--metal", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--jobs", type=int, default=6)
    p = sub.add_parser("baseline", help="Run uninstrumented measurements into a NEW directory")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--skip-metal", action="store_true")
    p = sub.add_parser("trace", help="Patch/build separate runtime after baseline, then record CPU operations")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--include-8k", action="store_true", help="Optional diagnostic; excluded from timing conclusions")
    p.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    for field in ("threads", "repetitions", "decode_steps", "jobs"):
        if hasattr(args, field) and getattr(args, field) <= 0:
            parser.error(f"--{field.replace(chr(95), chr(45))} must be positive")
    work = outside_recorded(args.work_dir)
    {"setup": setup, "baseline": baseline, "trace": trace}[args.command](args, work)


if __name__ == "__main__":
    main()
