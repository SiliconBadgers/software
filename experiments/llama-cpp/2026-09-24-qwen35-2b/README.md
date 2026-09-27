# QwenProfiling

llama.cpp is built locally, and Qwen3.5-2B computational graphs are captured and validated. Open `index.html` in a browser to browse them. This folder occupies approximately 3.8 GiB, mostly model weights.

This experiment was imported from `QwenProfiling` on 2026-09-24. See
[IMPORT.md](IMPORT.md) for provenance and local-only assets. The recorded
[2026-09-22 baseline](../2026-09-22/) remains a separate experiment.

## Interactive accelerator explorer

Open `index.html` (or `explorer/index.html`) for the local dashboard. It works directly from disk; no installation or server is needed. The original graph browser is now `graph-browser.html`.

The explorer exposes matrix/vector/recurrence/DMA/RISC-V counts and capacities, optional capabilities, integer precision packing, shared L1 SRAM banking and capacity, HBM, workload sizes, and editable resource costs. It provides live prefill/decode timing estimates, per-sequence and aggregate throughput, batch curves, operation-to-unit mappings, and Pareto sweeps. Designs export/import as JSON; sweep results export as CSV. The first default sweep contains 160 candidates.

These are uncalibrated analytical estimates. In particular, SRAM spill behavior and resource costs are approximations, and quantized accuracy is not validated. Read `explorer/MODEL.md` for formulas and calibration points.

```bash
# Optional local server, if preferred over opening the HTML file:
./scripts/serve-explorer.sh
# Model regression / sensitivity checks:
node scripts/test_model.cjs
# Regenerate the dashboard's graph data after a compatible recapture:
python3 scripts/export_explorer_data.py
```

## What is ready

- Unmodified llama.cpp checkout, pinned to `308883b335798865e73f8fee0d9a5fbfa13c8480`.
- CPU builds of `llama-completion`, `llama-simple`, and `llama-bench` under `llama.cpp/build/bin/`.
- Unsloth's BF16 GGUF conversion of Qwen3.5-2B. Its SHA-256 matches the publisher's file metadata; exact sources and revisions are in `manifest.json`.
- A standalone `bin/capture-graph` executable, using llama.cpp's public evaluation callback without modifying the checkout.
- Actual inference graphs for 128-token and 512-token prompts, followed by a single decode step after each prompt.
- Both prefill and decode graphs contain 1,441 scheduler nodes and 1,816 tensors including dependencies. Successful execution and finite logits were checked in all four captures.
- Raw graph JSON, complete Graphviz DOT, representative layer SVGs, operation CSVs, and summary reports.

## Open the graphs

Open `graph-browser.html`, then choose a workload. Each workload page offers prefill/decode and recurrent/full-attention layer diagrams. Full dependency graphs are available as DOT files. JSON is the authoritative machine-readable input for the accelerator model.

| Workload | Prompt tokens | Decode past tokens | Context capacity | Sequences |
| --- | ---: | ---: | ---: | ---: |
| graphs/pp128 | 128 | 128 | 512 | 1 |
| graphs/pp512 | 512 | 512 | 2,048 | 1 |

Each prefill emits logits only for the last prompt token, as normal generation does. Decode processes one token. Prompt token IDs are recorded in each `workload.json`.

## Reproduce

For a fresh clone, initialize the pinned dependency from the repository root:

```bash
git submodule update --init --recursive -- experiments/llama-cpp/2026-09-24-qwen35-2b/llama.cpp
cd experiments/llama-cpp/2026-09-24-qwen35-2b
```

Model weights and compiled binaries are local assets, excluded from Git. If the
model is missing, download the revision recorded in `manifest.json` and verify
it before building or capturing:

```bash
mkdir -p models
curl --fail --location --continue-at - \
  'https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-BF16.gguf' \
  --output models/Qwen3.5-2B-BF16.gguf
echo 'dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1  models/Qwen3.5-2B-BF16.gguf' | sha256sum --check
```

From this directory:

```bash
./scripts/build.sh
./scripts/capture.sh 128 512
./scripts/capture.sh 512 2048
```

The capture script overwrites the corresponding workload's generated files. To retain a separate run, supply an output directory:

```bash
./scripts/capture.sh 256 1024 ./graphs/pp256
```

Recompute reports from an existing graph without running the model:

```bash
python3 scripts/analyze_graph.py graphs/pp128
```

For ordinary text inference:

```bash
./llama.cpp/build/bin/llama-simple -m models/Qwen3.5-2B-BF16.gguf -ngl 0 -n 24 "The capital of France is"
```

Dependencies: C++17 compiler, CMake, Ninja, Python 3 standard library, and Graphviz (`dot`). The capture executable uses shared libraries from the local llama.cpp build via a relative runtime path. No Python packages, system installation, or running server are required. The current build uses CPU; CUDA is not enabled.

## Graph format

`prefill.json` and `decode.json` have:

- `metadata`: workload, model, context capacity, data type assumptions, and capture method.
- `scheduled_nodes`: exact order of nodes presented to the backend evaluation callback.
- `tensors`: unique tensor IDs, names, operation and subtype, dimensions, byte strides, element count, logical bytes, storage span, sources with operand slots, view source/offset, raw operation parameters, flags, and buffer metadata.

GGML shapes use the innermost dimension first. IDs are traversal identifiers, not topological ranks; use `scheduled_nodes` for schedule order. `op_params_i32` retains raw bit patterns and must be interpreted using the pinned GGML implementation. Aliases may share storage. Logical byte sizes and storage spans must not be summed as DRAM traffic or peak memory.

The callback snapshots metadata at `ask=true` and returns false, allowing normal execution without requesting tensor data readback. View dependencies are retained recursively. Graph validation checks valid references, acyclicity, tensor element counts, and scheduled dependency order. A finite-logit check is a smoke test, not a numerical equivalence test against another implementation.

## Scope for accelerator modeling

The captured text trunk contains 24 layers: 18 gated-delta recurrent layers and 6 full-attention layers. BF16 describes stored matrix weights; activations and many vector operations are F32, with F16 KV storage. Inspect per-tensor types for precision decisions.

Flash attention is disabled so attention matrix and softmax operations remain visible. GATED_DELTA_NET is retained as a fused GGML operation with its actual input/output shapes. It includes both sequence output and recurrent state in one tensor; its output dimensions must not be read as a simple token activation matrix. Attention shapes can include llama.cpp cache padding rather than only the number of live tokens.

Current arithmetic inventories count matrix and convolution MACs (two arithmetic operations per MAC). They deliberately do not claim total model FLOPs. Reductions, nonlinear functions, RoPE, and especially the fused gated-delta recurrence need additional hardware cost models. Memory traffic requires explicit tiling, SRAM residency, cache reuse, and transfer assumptions. Stateful aliases introduce read/write ordering requirements beyond ordinary tensor dataflow. CPU backend fusion and memory allocation are not an accelerator schedule.

This capture covers text inference only. It excludes the vision encoder, multi-token prediction, tokenizer, sampling, and host orchestration costs. The raw capture reports contain no accelerator timing estimate. The interactive explorer estimates latency and tokens/s using the hardware settings and explicit assumptions documented in `explorer/MODEL.md`.

## Hardware calibration inputs for the explorer

1. Processing-unit types and counts: matrix/MAC arrays, SIMD/vector units, scalar units, special functions, and any dedicated attention or recurrence support.
2. Clock rate, array dimensions or operations/cycle, supported precisions, and accumulation precision.
3. On-chip SRAM sizes and bandwidth, external-memory capacity/bandwidth, and interconnect topology/bandwidth if there are multiple cores.
4. Target workload: prompt length, generated length or decode context, sequence batch, and priority between prefill latency and decode tokens/s.

Partial specifications are fine; unspecified parameters can become explicit ranges in a later estimate. The explorer already maps graph operations to configurable units and estimates transfers and bottlenecks; these inputs improve its calibration.
