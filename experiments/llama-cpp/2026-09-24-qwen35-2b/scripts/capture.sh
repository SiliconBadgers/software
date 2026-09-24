#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PROMPT_TOKENS=${1:-128}
CONTEXT=${2:-512}
OUTDIR=${3:-"$ROOT/graphs/pp$PROMPT_TOKENS"}
"$ROOT/bin/capture-graph" "$ROOT/models/Qwen3.5-2B-BF16.gguf" "$OUTDIR" "$PROMPT_TOKENS" "$CONTEXT"
python3 "$ROOT/scripts/analyze_graph.py" "$OUTDIR"
