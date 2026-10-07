"""Shared helpers: command running with --dry-run, tool lookup, hashing, deterministic gzip."""
from dataclasses import dataclass, field
import gzip
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

CORE = Path(__file__).resolve().parents[1]
REPO = CORE.parent
sys.path.insert(0, str(CORE / "analysis"))


@dataclass
class Context:
    experiment: dict
    platform: dict
    os_label: str
    work: Path
    run_dir: Path
    lengths: list
    threads: int
    repetitions: int
    trace_repetitions: int
    decode_steps: int
    graph_lengths: list
    jobs: int = 6
    metal: str = "auto"              # auto | on | off
    skip_metal: bool = False
    cuda: str = "off"                # off | on (CPU reference + CUDA) | only (fast exploratory run)
    cuda_controls: bool = False
    cuda_no_host_control: bool = False
    run_trace: bool = True
    run_graphs: bool = True
    include_diagnostic: bool = False
    model_override: Path = None
    dry_run: bool = False
    smoke: bool = False
    run_prompt_graphs: bool = True
    prompts_only: bool = False
    run_prompt_timings: bool = True
    prompt_set: Path = REPO / "core" / "prompts" / "prompts.json"
    state: dict = field(default_factory=dict)

    def run(self, command, **kwargs):
        command = [str(arg) for arg in command]
        print("+", shlex.join(command), flush=True)
        if self.dry_run:
            return None
        return subprocess.run(command, check=True, **kwargs)


def tool(name):
    """Prefer tools installed in the active virtual environment (cmake, ninja) over system ones."""
    exe = name + (".exe" if os.name == "nt" else "")
    local = Path(sys.executable).parent / exe
    found = str(local) if local.is_file() else shutil.which(name)
    if not found:
        raise RuntimeError(f"Missing {name}; install core/requirements.txt (the unix/ and windows/ wrappers do this)")
    return found


def sha256_of(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gzip_in_place(path):
    """Compress with a fixed timestamp so identical inputs give identical archives."""
    path = Path(path)
    target = Path(str(path) + ".gz")
    with path.open("rb") as src, target.open("wb") as raw:
        with gzip.GzipFile(filename=path.name, mode="wb", fileobj=raw, mtime=0) as out:
            shutil.copyfileobj(src, out)
    path.unlink()
    return target


def exe_path(ctx, build_dir, name):
    return Path(build_dir) / "bin" / (name + ctx.platform["exe_suffix"])
