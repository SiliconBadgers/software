#!/usr/bin/env bash
# Linux and macOS entry point. uv manages .venv from the committed lockfile.
#   unix/run.sh doctor
#   unix/run.sh profile [--smoke] [--dry-run] ...
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
UV=${UV:-uv}

command -v "$UV" >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}

# --project preserves the caller's working directory and relative CLI arguments.
exec "$UV" run --locked --project "$ROOT" --group profiling \
  --python "${PYTHON:-3.12}" python "$ROOT/core/runner/cli.py" \
  --platform-dir "$ROOT/unix" "$@"
