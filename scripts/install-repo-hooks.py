#!/usr/bin/env python3
"""Opt this clone into the tracked commit guard without replacing other hooks."""
import os
from pathlib import Path
import subprocess
import sys


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    root = Path(git("rev-parse", "--show-toplevel"))
    os.chdir(root)
    configured = subprocess.run(["git", "config", "--get", "core.hooksPath"], capture_output=True, text=True)
    if configured.returncode not in (0, 1):
        sys.exit(configured.stderr)
    if configured.returncode == 0:
        path = Path(configured.stdout.strip()).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.resolve() != (root / ".githooks").resolve():
            sys.exit("Existing core.hooksPath preserved. Ask the maintainer to compose the hooks before changing it.")
    else:
        hooks = Path(git("rev-parse", "--git-path", "hooks"))
        active = [p.name for p in hooks.glob("*") if p.is_file() and not p.name.endswith(".sample") and os.access(p, os.X_OK)]
        if active:
            sys.exit("Existing Git hooks preserved: " + ", ".join(active) + ". Ask the maintainer to compose the hooks.")
    hook = root / ".githooks/pre-commit"
    if not hook.is_file() or not os.access(hook, os.X_OK):
        sys.exit("Tracked .githooks/pre-commit is missing or not executable. Restore its executable bit first.")
    git("config", "--local", "core.hooksPath", ".githooks")
    print("Installed local commit guard. Linked worktrees share this clone's configuration.")


if __name__ == "__main__":
    main()
