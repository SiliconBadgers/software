"""Pinned llama.cpp checkout and model download, both idempotent and verified."""
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

from common import sha256_of


def state_path(ctx):
    return ctx.work / "state.json"


def load_state(ctx):
    path = state_path(ctx)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def save_state(ctx, state):
    ctx.work.mkdir(parents=True, exist_ok=True)
    state_path(ctx).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _head(source):
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


def ensure_source(ctx):
    """Check out the pinned commit into work/llama.cpp (never resets an existing checkout)."""
    pin = ctx.experiment["llama_cpp"]
    source = ctx.work / "llama.cpp"
    if source.exists():
        if _head(source) != pin["commit"]:
            raise ValueError(f"{source} is not at pinned commit {pin['commit']}; use a fresh --work-dir")
        return source
    ctx.run(["git", "init", source])
    # Git for Windows commonly sets core.autocrlf=true globally. Force LF so the trace patch applies.
    ctx.run(["git", "-C", source, "config", "core.autocrlf", "false"])
    ctx.run(["git", "-C", source, "config", "core.eol", "lf"])
    ctx.run(["git", "-C", source, "remote", "add", "origin", pin["repository"]])
    ctx.run(["git", "-C", source, "fetch", "--depth=1", "origin", pin["commit"]])
    ctx.run(["git", "-C", source, "checkout", "--detach", pin["commit"]])
    return source


def verify_model(path, model):
    path = Path(path)
    if path.stat().st_size != model["size_bytes"] or sha256_of(path) != model["sha256"]:
        raise ValueError(f"Model size/SHA256 mismatch: {path}")


def ensure_model(ctx):
    """Return the verified model path, downloading the pinned revision if needed."""
    model = ctx.experiment["model"]
    path = (ctx.model_override.expanduser().resolve() if ctx.model_override
            else ctx.work / "models" / model["file"])
    state = load_state(ctx)
    stamp = None
    if path.is_file():
        stamp = {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        if state.get("model_verified") == stamp:
            return path
        print("Verifying model size and SHA256 ...", flush=True)
        if not ctx.dry_run:
            verify_model(path, model)
            state["model_verified"] = stamp
            save_state(ctx, state)
        return path
    if ctx.model_override:
        raise FileNotFoundError(path)
    url = f'https://huggingface.co/{model["repo"]}/resolve/{model["revision"]}/{model["file"]}'
    print("Downloading pinned model:", url, flush=True)
    if ctx.dry_run:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    verify_model(partial, model)
    partial.rename(path)
    stamp = {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    state["model_verified"] = stamp
    save_state(ctx, state)
    return path
