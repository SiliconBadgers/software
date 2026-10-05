"""Choose the unix/ or windows/ platform profile and derive the OS label used in run folder names."""
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from common import REPO


def load(platform_dir=None):
    """Returns (profile dict, os_label). platform_dir defaults from the running OS."""
    directory = Path(platform_dir) if platform_dir else REPO / ("windows" if sys.platform == "win32" else "unix")
    if not directory.is_absolute():
        directory = REPO / directory
    profile = json.loads((directory / "platform.json").read_text(encoding="utf-8"))
    label = profile["os_labels"].get(sys.platform)
    if label is None:
        raise RuntimeError(f"{directory.name}/platform.json has no os_labels entry for '{sys.platform}'")
    profile["directory"] = str(directory)
    return profile, label


def metal_available(profile, os_label):
    return os_label in profile["metal_os_labels"] and platform.machine().lower() in ("arm64", "aarch64")


def cuda_compiler():
    """Find nvcc without requiring a newly installed toolkit to be on this process's PATH."""
    executable = "nvcc.exe" if os.name == "nt" else "nvcc"
    roots = [value for value in (os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME")) if value]
    candidates = [Path(root) / "bin" / executable for root in roots]
    if os.name == "nt":
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if base.is_dir():
            candidates += sorted(base.glob("v*/bin/nvcc.exe"), reverse=True)
    else:
        candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    stable = next((str(path) for path in candidates if path.is_file()), None)
    return stable or shutil.which("nvcc")
