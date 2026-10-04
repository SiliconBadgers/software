#!/usr/bin/env bash
# Linux and macOS entry point. Creates .venv on first use, then runs the core runner.
#   unix/run.sh doctor
#   unix/run.sh profile [--smoke] [--dry-run] ...
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VENV="$ROOT/.venv"
PY=${PYTHON:-python3}

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
    || { echo "Python 3.12 or newer is required (set PYTHON to choose one)"; exit 1; }
  "$PY" -m venv "$VENV"
fi

want=$("$VENV/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$ROOT/core/requirements.txt")
have=$(cat "$VENV/.sb-requirements.sha256" 2>/dev/null || true)
if [ "$want" != "$have" ]; then
  "$VENV/bin/python" -m pip install -r "$ROOT/core/requirements.txt"
  printf '%s' "$want" > "$VENV/.sb-requirements.sha256"
fi

exec "$VENV/bin/python" "$ROOT/core/runner/cli.py" --platform-dir "$ROOT/unix" "$@"
