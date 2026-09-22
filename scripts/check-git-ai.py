#!/usr/bin/env python3
"""Check local Git AI installation/configuration; actual capture needs a smoke test."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tomllib


def main():
    errors = []
    executable = shutil.which("git-ai") or str(Path.home() / ".git-ai/bin/git-ai")
    if not Path(executable).is_file():
        print("FAIL: Git AI missing. Run bash scripts/setup-git-ai.sh.")
        return 1
    codex_config = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    try:
        config = tomllib.loads(codex_config.read_text())
        if not config.get("features", {}).get("hooks"):
            errors.append("Codex features.hooks is not enabled")
        hooks = config.get("hooks", {})
        for event in ("PreToolUse", "PostToolUse", "Stop"):
            found = False
            event_key = re.sub(r"(?<!^)(?=[A-Z])", "_", event).lower()
            for group_index, group in enumerate(hooks.get(event, [])):
                for hook_index, hook in enumerate(group.get("hooks", [])):
                    command = shlex.split(hook.get("command", ""))
                    if not command or Path(command[0]).name != "git-ai":
                        continue
                    if command[1:] != ["checkpoint", "codex", "--hook-input", "stdin"]:
                        continue
                    key = f"{codex_config}:{event_key}:{group_index}:{hook_index}"
                    if hooks.get("state", {}).get(key, {}).get("enabled") is not False:
                        found = True
            if not found:
                errors.append(f"Codex {event} Git AI hook is missing or disabled")
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot read Codex configuration: {exc}")
    try:
        local_config = json.loads((Path.home() / ".git-ai/config.json").read_text())
        if local_config.get("prompt_storage") != "local":
            errors.append("Set git-ai config set prompt_storage local")
        if local_config.get("telemetry_oss") != "off":
            errors.append("Set git-ai config set telemetry_oss off")
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot read Git AI configuration: {exc}")
    for args in (("--version",), ("bg", "status")):
        try:
            result = subprocess.run([executable, *args], capture_output=True, text=True, timeout=15)
            if result.returncode:
                errors.append(result.stderr.strip() or f"git-ai {' '.join(args)} failed")
            elif args == ("bg", "status"):
                status = json.loads(result.stdout)
                if not status.get("ok") or status.get("daemon_running") is False or not status.get("daemon"):
                    errors.append("Git AI daemon is not ready")
                if status.get("data", {}).get("last_error"):
                    errors.append(f"Git AI repository error: {status['data']['last_error']}")
            else:
                print("Git AI version:", result.stdout.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
    for error in errors:
        print("FAIL:", error)
    if not errors:
        print("PASS: configuration and service checks. Restart Codex after first setup.")
        print("Confirm actual capture with the smoke test in docs/git-ai.md.")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
