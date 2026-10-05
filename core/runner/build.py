"""Configure/build the harness against the pinned llama.cpp, and apply the CPU trace patch."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

from common import CORE, exe_path, tool
import fetch

HARNESS = CORE / "harness"
PATCH = HARNESS / "sb-cpu-profile.patch"
BASELINE_TARGETS = ["sb-profile", "sb-profile-prompt", "sb-capture-graph"]
PROFILE_TARGETS = ["sb-profile", "sb-profile-prompt"]


def find_compilers(ctx):
    """Returns {"c": path|None, "cxx": path|None}. Env SB_CC / SB_CXX win, then the platform list."""
    found = {}
    for key, env in (("c", "SB_CC"), ("cxx", "SB_CXX"), ("rc", "SB_RC")):
        listed = ctx.platform["compilers"].get(key, [])
        candidates = ([os.environ[env]] if os.environ.get(env) else []) + listed
        found[key] = next((c for c in map(_resolve, candidates) if c), None)
        if listed and not found[key]:
            raise RuntimeError(f"No {key.upper()} compiler found; tried {candidates}. Set {env} to a path.")
    return found


def _resolve(candidate):
    """A file path, a glob pattern (newest match wins, e.g. the latest Windows SDK) or a name on PATH."""
    path = Path(candidate)
    if path.is_file():
        return str(path)
    if any(ch in candidate for ch in "*?["):
        anchor = Path(path.anchor) if path.anchor else Path(".")
        matches = sorted(m for m in anchor.glob(str(path.relative_to(anchor)) if path.anchor else candidate) if m.is_file())
        return str(matches[-1]) if matches else None
    return shutil.which(candidate)


def build_dir(ctx, variant):
    return ctx.work / ("build-" + variant)


def source_dir(ctx, variant):
    """The baseline build reads the pristine checkout; the traced build reads its own patched worktree.
    They must never share a source tree: a later rebuild of the baseline would otherwise silently
    compile the instrumentation into it."""
    return ctx.work / ("llama.cpp" if variant == "baseline" else "llama.cpp-profile")


def build_variant(ctx, variant, targets, compile=True):
    """variant: 'baseline' (pristine source) or 'profile' (patched source). compile=False only configures."""
    source = source_dir(ctx, variant)
    directory = build_dir(ctx, variant)
    compilers = find_compilers(ctx)
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    # CMake writes these values into generated files verbatim, so Windows backslashes would be read as escapes.
    use_cuda = bool(ctx.state.get("cuda")) and variant == "baseline"
    configure = [tool("cmake"), "-S", HARNESS, "-B", directory, "-G", ctx.platform["cmake_generator"],
                 "-DCMAKE_BUILD_TYPE=Release", f"-DLLAMA_CPP_SOURCE={Path(source).as_posix()}", "-DGGML_NATIVE=ON",
                 "-DGGML_METAL=" + ("ON" if ctx.state.get("metal") else "OFF"),
                 "-DGGML_CUDA=" + ("ON" if use_cuda else "OFF")]
    if use_cuda:
        import platform_profile
        configure.append(f"-DCMAKE_CUDA_COMPILER={Path(platform_profile.cuda_compiler()).as_posix()}")
    if compilers["c"]:
        configure.append(f"-DCMAKE_C_COMPILER={Path(compilers['c']).as_posix()}")
    if compilers["cxx"]:
        configure.append(f"-DCMAKE_CXX_COMPILER={Path(compilers['cxx']).as_posix()}")
    if compilers["rc"]:
        configure.append(f"-DCMAKE_RC_COMPILER={Path(compilers['rc']).as_posix()}")
    configure += ctx.platform["cmake_args"]
    ctx.run(configure, env=env)
    if not compile:
        return compilers
    ctx.run([tool("cmake"), "--build", directory, "--target", *targets, "-j", str(ctx.jobs)], env=env)
    return compilers


def write_lf_patch(ctx):
    """Write an LF-only copy of the trace patch. A CRLF working-tree copy (Git for Windows autocrlf)
    would not apply to the LF llama.cpp checkout."""
    lf_patch = ctx.work / "sb-cpu-profile.lf.patch"
    if not ctx.dry_run:
        ctx.work.mkdir(parents=True, exist_ok=True)
        lf_patch.write_bytes(PATCH.read_bytes().replace(b"\r\n", b"\n"))
    return lf_patch


def build_config(ctx, variant="baseline"):
    """Requested CMake options and what the ninja build actually enables (e.g. OpenMP is often requested but absent)."""
    directory = build_dir(ctx, variant)
    cache, ninja = directory / "CMakeCache.txt", directory / "build.ninja"
    if not cache.is_file() or not ninja.is_file():
        return None
    wanted = ("CMAKE_BUILD_TYPE", "BUILD_SHARED_LIBS", "GGML_NATIVE", "GGML_OPENMP", "GGML_METAL", "GGML_CUDA",
              "CMAKE_CUDA_COMPILER", "GGML_LLAMAFILE",
              "GGML_CPU_REPACK", "GGML_BACKEND_DL")
    requested = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        key, _, rest = line.partition(":")
        if key in wanted and "=" in rest:
            requested[key] = rest.split("=", 1)[1]
    text = ninja.read_text(encoding="utf-8", errors="replace")
    return {"requested": requested,
            "baseline_build_has_trace_code": has_trace_code(ctx, "baseline"),
            "traced_build_has_trace_code": has_trace_code(ctx, "profile") if build_dir(ctx, "profile").is_dir() else None,
            "effective": {"openmp": "GGML_USE_OPENMP" in text, "llamafile": "GGML_USE_LLAMAFILE" in text,
                          "march_native": "-march=native" in text, "metal": "GGML_USE_METAL" in text,
                          "cuda": "GGML_USE_CUDA" in text}}


def _is_clean(source):
    return not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()


def apply_patch(ctx):
    """Create the patched worktree next to the pristine checkout (once) and apply sb-cpu-profile.patch to it."""
    pristine, patched = source_dir(ctx, "baseline"), source_dir(ctx, "profile")
    state = fetch.load_state(ctx)
    if state.get("profile_patched") and patched.is_dir():
        return patched
    if not patched.exists():
        ctx.run(["git", "-C", pristine, "worktree", "add", "--detach", patched])
    if not ctx.dry_run and not _is_clean(patched):
        raise ValueError(f"{patched} already has modifications; delete it (git worktree remove --force) and retry")
    lf_patch = write_lf_patch(ctx)
    ctx.run(["git", "-C", patched, "apply", "--check", lf_patch])
    ctx.run(["git", "-C", patched, "apply", lf_patch])
    if not ctx.dry_run:
        state = fetch.load_state(ctx)
        state["profile_patched"] = True
        fetch.save_state(ctx, state)
    return patched


TRACE_MARKER = b"SB_CPU_TRACE"   # environment variable read only by the trace patch


def has_trace_code(ctx, variant):
    """True if the built CPU backend contains the trace patch. Looks in the shared library, or in every
    binary of the build if the CPU backend is linked statically."""
    bin_dir = build_dir(ctx, variant) / "bin"
    candidates = [p for p in bin_dir.iterdir() if p.is_file() and "ggml-cpu" in p.name] if bin_dir.is_dir() else []
    candidates = candidates or [p for p in bin_dir.iterdir() if p.is_file()] if bin_dir.is_dir() else []
    return any(TRACE_MARKER in p.read_bytes() for p in candidates)


def check_instrumentation(ctx, variant, expected):
    """Fail loudly if the pristine build is instrumented or the traced build is not."""
    if ctx.dry_run:
        return
    found = has_trace_code(ctx, variant)
    if found != expected:
        raise ValueError(f"build-{variant} {'contains' if found else 'lacks'} the trace patch but should "
                         f"{'contain' if expected else 'not contain'} it; delete work/build-{variant} and re-run")


def ensure_baseline(ctx):
    """Pristine build: sb-profile and sb-capture-graph. Always runs the (incremental) build, so edits to the
    harness sources are picked up; ninja does nothing when everything is current."""
    pristine = source_dir(ctx, "baseline")
    if not ctx.dry_run and pristine.is_dir() and not _is_clean(pristine):
        raise ValueError(f"{pristine} is modified. The pristine checkout must stay unpatched; the traced build "
                         "uses llama.cpp-profile. Restore it with: git -C <that path> checkout -- .")
    compilers = build_variant(ctx, "baseline", BASELINE_TARGETS)
    check_instrumentation(ctx, "baseline", expected=False)
    if not ctx.dry_run:
        state = fetch.load_state(ctx)
        state.update(baseline_ready=True, metal=bool(ctx.state.get("metal")), cuda=bool(ctx.state.get("cuda")))
        fetch.save_state(ctx, state)
    return compilers


def ensure_profile(ctx):
    """Patched build for op tracing, created after baselines exist. Incremental like the baseline build."""
    state = fetch.load_state(ctx)
    if not state.get("baseline_ready") and not ctx.dry_run:
        raise ValueError("Build and run the pristine baseline before patching")
    apply_patch(ctx)
    build_variant(ctx, "profile", PROFILE_TARGETS)
    check_instrumentation(ctx, "profile", expected=True)
    if not ctx.dry_run:
        state = fetch.load_state(ctx)
        state["profile_ready"] = True
        fetch.save_state(ctx, state)
