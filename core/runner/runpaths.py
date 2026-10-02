"""Run folders: results/<YYYY-MM-DD>_<HHMM>_<os>[-n]/. Never reused, never inside experiments/."""
from datetime import datetime
from pathlib import Path

from common import REPO

PROTECTED = REPO / "experiments"


def assert_writable(path):
    path = Path(path).resolve()
    if path == PROTECTED.resolve() or PROTECTED.resolve() in path.parents:
        raise ValueError("experiments/ holds recorded evidence and is immutable; runs belong in results/")


def run_name(os_label, now=None):
    now = now or datetime.now()
    return f"{now:%Y-%m-%d}_{now:%H%M}_{os_label}"


def create_run_dir(results_root, os_label, now=None, create=True):
    """Return a fresh run directory; a same-minute collision gets -2, -3 ... (nothing is overwritten)."""
    root = Path(results_root).resolve()
    assert_writable(root)
    base = run_name(os_label, now)
    candidate, n = root / base, 1
    while candidate.exists():
        n += 1
        candidate = root / f"{base}-{n}"
    if create:
        candidate.mkdir(parents=True, exist_ok=False)
    return candidate
