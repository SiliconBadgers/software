"""Choose the unix/ or windows/ platform profile and derive the OS label used in run folder names."""
import json
from pathlib import Path
import platform
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
