# CPU operation profiler

Prepared against llama.cpp f46bc30cb6a7f68a67e34a00061e20a4ad1eff43. The patch was applied locally after collecting unpatched baselines, rebuilt successfully, and exercised by the saved traces. No upstream changes were submitted.

## Applying and rebuilding

The [reproduction runner](reproduce.py) collects pristine baselines, then applies the patch and builds a separate instrumented library. For manual application, start from the experiment directory after setup and baseline collection:

```sh
git -C work/llama.cpp apply --check ../../sb-cpu-profile.patch
git -C work/llama.cpp apply ../../sb-cpu-profile.patch
PATH="$PWD/.venv/bin:$PATH" .venv/bin/cmake -S work/llama.cpp -B work/build-profile -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=ON -DGGML_METAL=ON
.venv/bin/cmake --build work/build-profile --target llama-bench llama-simple -j 6
clang++ -O3 -std=c++17 profile.cpp -I work/llama.cpp/include -I work/llama.cpp/ggml/include -L work/build-profile/bin -Wl,-rpath,"$PWD/work/build-profile/bin" -lllama -lggml -lggml-base -lggml-cpu -o work/profile-profile
```

Use `-DGGML_METAL=OFF` and the host C++ compiler for CPU-only hosts. Check applicability first and never apply the patch twice. The instrumented harness links only `work/build-profile/bin`; the pristine baseline binary continues to link `work/build-baseline/bin`. The authoritative source change is `sb-cpu-profile.patch`; no complete upstream source copy is included in this package.

Set `SB_CPU_TRACE` to an absolute JSONL output path. The logger appends, so use a fresh path per run. The harness sets `SB_CPU_PHASE` before each synchronized decode call, for example `p128_r0_prefill` or `p128_r0_decode0`. `warmup*` and exactly `idle` phases are skipped; an unset phase is labeled `unspecified`. Do not mutate environment variables concurrently with graph execution. One harness/context should write a given path; concurrent file writers are not supported. There is no evaluation callback, no forced graph splitting, no additional worker barrier, and no modification to CPU fusion decisions.

## What each record means

Thread 0 takes a microsecond timestamp immediately before fused-op detection/computation and another after the existing per-node completion barrier. The final graph node does not normally have a per-node barrier, so its ending timestamp is deferred until after the existing final graph barrier. Fused RMSNorm + MUL remains one interval and the successor is identified separately. Nodes skipped by the normal graph scheduler are omitted. Some dispatched nodes still have zero-sized dimensions (for example, output-head placeholders in earlier prefill microbatches); they do not represent nonzero arithmetic or data movement. The records are allocated once before releasing workers and store only node index, fusion length and elapsed time during execution. Tensor metadata and JSON serialization occur after every worker has completed that graph, while all graph tensors remain valid.

Each JSONL record contains `graph_id`, `phase`, `node_index`, `n_threads`, `graph_status`, `elapsed_us`, a `tensor` descriptor (`name`, `op`, `dtype`, `ne`, `nbytes`, `is_view`), the complete positional `src` descriptor array with null slots, and `fused_successors` descriptors. `nbytes` is logical tensor size, not measured memory traffic; views can alias other storage. The graph ID counts traced graphs within this process. Read source 0's type/shape for matrix weights: an output tensor's dtype does not describe its quantized weights. GGML dimensions are in `ne[0]`-first order. `fused_successors` does not carry its own source-array metadata; consult model/graph definitions for the fused multiply weight.

## Interpretation and limitations

These are synchronized CPU wall-clock intervals, not thread CPU time, hardware cycles, or FPGA/ASIC performance estimates. They include kernel computation, any internal synchronization, the regular completion barrier, scheduler/OS stalls, and fusion dispatch. They do not distinguish arithmetic from dequantization, memory service, or thread synchronization within a kernel. The optional abort callback is also within the interval. Timing a fused pair does not separate its constituent costs.

No extra start barrier is inserted. Workers can advance slightly before thread 0 timestamps a node (especially at the start of a graph or while thread 0 records the preceding interval). Therefore small-node intervals may undercount some worker activity and recording overhead can perturb synchronization. The timer's microsecond resolution makes tiny node times noisy, including zero. Aggregate many invocations and retain honest precision. Thread-0 record writes and loop bookkeeping between nodes are not included in each node's interval. The existing extra final barrier after a last compute node followed by empty nodes is not charged to that node. Thus the sum of intervals need not exactly equal graph wall time.

Allocation and JSON output are excluded from individual operation intervals, but they ARE inside the enclosing `ggml_graph_compute`/`llama_decode` call. End-to-end throughput from an enabled-trace run is consequently not a valid primary benchmark. Use the already collected uninstrumented CPU/Metal baselines for throughput; compare timing sums and dominant-operation rankings across repeated traces to assess stability. Trace results describe the CPU path only. They do not profile Metal kernel time and should not be combined with Metal throughput as if one execution path produced both.

Validation passed: patch applicability, build, smoke trace, successful graph statuses/nonnegative times, expected phase and operation coverage, and bit-identical saved CPU logits after prefill and final decode for all four context lengths. See the report for the separate 8K timing anomaly; correct outputs do not guarantee trustworthy timings.
