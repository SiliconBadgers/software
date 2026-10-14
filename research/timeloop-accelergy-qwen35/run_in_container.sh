#!/usr/bin/env bash
set -euo pipefail

# Run from Ubuntu/WSL with the official tutorial container already running.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
container_name="${TIMELOOP_CONTAINER:-timeloop-accelergy-exercises-tutorial-1}"
container_dir=/home/workspace/siliconbadgers-bpu

docker inspect --format '{{.State.Running}}' "$container_name" | grep -qx true
docker exec "$container_name" mkdir -p \
  "$container_dir/arch" "$container_dir/mapping" "$container_dir/problem"
for folder in arch mapping problem; do
  docker cp "$project_dir/inputs/$folder/." "$container_name:$container_dir/$folder/"
done
docker cp "$project_dir/run_baseline_timeloop.py" "$container_name:$container_dir/"
docker cp "$project_dir/run_qwen_matmuls.py" "$container_name:$container_dir/"

docker exec -w "$container_dir" "$container_name" python3 run_baseline_timeloop.py
docker exec -w "$container_dir" "$container_name" python3 run_qwen_matmuls.py

# Docker's chmod preservation can fail when docker cp targets /mnt/c directly.
scratch_dir="$(mktemp -d)"
docker cp "$container_name:$container_dir/output/corrected-2048/decode" "$scratch_dir/"
docker cp "$container_name:$container_dir/output/corrected-2048/prefill" "$scratch_dir/"
mkdir -p "$project_dir/results/timeloop-run/decode" "$project_dir/results/timeloop-run/prefill"
cp -r "$scratch_dir/decode/." "$project_dir/results/timeloop-run/decode/"
cp -r "$scratch_dir/prefill/." "$project_dir/results/timeloop-run/prefill/"

for case_name in \
  ffn-gate-decode-a-stationary \
  ffn-gate-prefill-a-stationary \
  ffn-gate-prefill-weight-reuse \
  ffn-out-decode-a-stationary \
  ffn-out-prefill-a-stationary \
  ffn-out-prefill-weight-reuse \
  output-head-a-stationary; do
  docker cp "$container_name:$container_dir/output/qwen-matmuls/$case_name" "$scratch_dir/"
  mkdir -p "$project_dir/results/qwen-ffn-2026-10-12/$case_name"
  cp -r "$scratch_dir/$case_name/." "$project_dir/results/qwen-ffn-2026-10-12/$case_name/"
done

python3 "$project_dir/summarize_timeloop.py"
printf 'Generated reports copied to %s/results\n' "$project_dir"
printf 'Temporary transfer copies remain at %s\n' "$scratch_dir"
