#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cmake -S "$ROOT/llama.cpp" -B "$ROOT/llama.cpp/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$ROOT/llama.cpp/build" --parallel 6 --target llama-completion llama-bench llama-simple
mkdir -p "$ROOT/bin"
c++ -std=c++17 -O2 "$ROOT/src/capture_graph.cpp" \
  -I"$ROOT/llama.cpp/include" -I"$ROOT/llama.cpp/ggml/include" -I"$ROOT/llama.cpp/vendor" \
  -L"$ROOT/llama.cpp/build/bin" -Wl,-rpath,'$ORIGIN/../llama.cpp/build/bin' \
  -lllama -lggml -lggml-base -o "$ROOT/bin/capture-graph"
