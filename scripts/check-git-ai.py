#!/usr/bin/env python3
"""Validate the supported local Codex/Git AI setup, without claiming capture."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

try:
    import tomllib
except ImportError:
    sys.exit("FAIL: Python 3.11+ is required for the Git AI setup check.")


def run(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"{' '.join(args)} failed")
    return result.stdout.strip()


def check_hooks(config, config_path):
    errors = []
    if not config.get("features", {}).get("hooks"):
        errors.append("Codex features.hooks is not enabled")
    hooks = config.get("hooks", {})
    for event in ("PreToolUse", "PostToolUse", "Stop"):
        found = False
        event_key = re.sub(r"(?<!^)(?=[A-Z])", "_", event).lower()
        for group_index, group in enumerate(hooks.get(event, [])):
            # A restricted matcher may silently miss an editing tool.
            if group.get("matcher", "*") not in ("", "*"):
                continue
            for hook_index, hook in enumerate(group.get("hooks", [])):
                command = shlex.split(hook.get("command", ""))
                if not command or Path(command[0]).name != "git-ai":
                    continue
                if command[1:] != ["checkpoint", "codex", "--hook-input", "stdin"]:
                    continue
                key = f"{config_path}:{event_key}:{group_index}:{hook_index}"
                binary = shutil.which(command[0]) if not Path(command[0]).is_absolute() else command[0]
                if (binary and os.access(binary, os.X_OK)
                        and hooks.get("state", {}).get(key, {}).get("enabled") is not False):
                    found = True
        if not found:
            errors.append(f"Codex {event} Git AI hook is missing, restricted or disabled")
    return errors


def main():
    errors = []
    try:
        root = Path(run("git", "rev-parse", "--show-toplevel"))
        hooks_path = run("git", "config", "--get", "core.hooksPath")
        resolved = Path(hooks_path).expanduser()
        if not resolved.is_absolute():
            resolved = root / resolved
        if resolved.resolve() != (root / ".githooks").resolve():
            errors.append("Repository guard inactive. Run python3 scripts/install-repo-hooks.py")
        if not os.access(root / ".githooks/pre-commit", os.X_OK):
            errors.append("Repository pre-commit hook is missing or not executable")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        errors.append(f"Repository hook setup: {exc}. Run python3 scripts/install-repo-hooks.py")
    executable = shutil.which("git-ai")
    if not executable:
        errors.append("git-ai is not on PATH. Run bash scripts/setup-git-ai.sh and reopen your terminal")
    config_path = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    try:
        errors.extend(check_hooks(tomllib.loads(config_path.read_text()), config_path))
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot validate Codex config: {exc}")
    try:
        config = json.loads((Path.home() / ".git-ai/config.json").read_text())
        for key, value in (("prompt_storage", "local"), ("telemetry_oss", "off")):
            if config.get(key) != value:
                errors.append(f"Run git-ai config set {key} {value}")
        # This rollout supports the simple global local-storage policy only.
        # Overrides need an explicit review, not a guess about precedence.
        for key in ("include_prompts_in_repositories", "allow_repositories", "exclude_repositories"):
            if config.get(key):
                errors.append(f"Git AI {key} overrides need review before this check can pass")
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot validate Git AI config: {exc}")
    if executable:
        try:
            version = run(executable, "--version")
            if version.split()[-1] != "1.7.5":
                errors.append(f"Untested Git AI version {version}; this setup supports 1.7.5")
            status = json.loads(run(executable, "bg", "status"))
            daemon = status.get("daemon", {})
            if not status.get("ok") or status.get("daemon_running") is False or not daemon:
                errors.append("Git AI daemon is not ready. Check git-ai bg status from a normal terminal")
            if daemon.get("sequencer_stalled") or daemon.get("snapshot_partial"):
                errors.append("Git AI daemon is stalled or its health snapshot is incomplete")
            if status.get("data", {}).get("last_error"):
                # This field is a retained error, not a current health status.
                print("NOTE: Git AI retains a repository sync error. Inspect git-ai bg status; "
                      "reconcile notes with git-ai fetch-notes origin and verify publication after pushing.")
            capture = json.loads(run(executable, "status", "--json", "--diff-only"))
            if not isinstance(capture.get("stats"), dict):
                errors.append("Git AI did not return readable working-change attribution")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            errors.append(f"Git AI service check: {exc}")
    for error in errors:
        print("FAIL:", error)
    if not errors:
        print("PASS: Codex/Git AI configuration, service and local commit guard.")
        print("Check git-ai status before committing and git-ai stats HEAD afterward; setup alone is not capture proof.")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
