#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PORT=${1:-8765}
printf 'Open http://127.0.0.1:%s/explorer/index.html\n' "$PORT"
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$ROOT"
