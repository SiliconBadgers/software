# Validation record

What was checked, how, what came out, and what is **not** validated. Nothing here validates the full accelerator: the
checks establish that the engine's *accounting* is right and that it reproduces the explorer it replaces. The accelerator
timing equations are analytical estimates until RTL or synthesis numbers replace the placeholders in `ASSUMPTIONS.md`. No
licensed-tool check was run.

Run `2026-09-27_1839_windows-2`: llama.cpp `f46bc30`, `Qwen3.5-2B-Q4_K_M.gguf` (sha256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`),
Windows 11, i7-13620H, 6 threads. Engine 0.1.0, Python 3.13.14, numpy 2.5.3. Explorer reference numbers: `engine.js` at origin/main
`11d36ed` under Node 24.14.0.

## 1. Summary

| Check | Command / test | Result |
|---|---|---|
| MAC accounting vs every capture summary | `python -m sbengine parity` | all match: 22 workloads x 2 phases, `MUL_MAT` and `SSM_CONV` exact |
| Fused attention does the same work as the unfused capture | `test_fused_attention_does_the_same_matmul_work_as_the_unfused_capture` | exact, pp512 and pp8192, both phases |
| Reproduces the explorer with its equations restored | `test_legacy_switches_reproduce_the_explorer_on_its_own_graphs` | prefill <= 0.001 %, decode +0.46 % to +0.99 % (8 points) |
| Accounting predicts measured CPU time, held out | `python -m sbengine validate-cpu` | prefill mean 2.2 % (max 4.5 %), decode mean 7.1 % (max 16.4 %) |
| Explorer's own regression suite (independent) | `node scripts/test_model.cjs` on the explorer | 23 checks pass |
| Engine unit and invariant tests | `python -m unittest discover -s tests` | 52 tests pass |

## 2. Reproducing the explorer

`profiles/legacy-equations.json` sets every switchable equation back to the explorer's version. Compared with `engine.js`
on the explorer's own fa-off graphs, both saved designs:

| Design | Tokens | Phase | Explorer | Engine (legacy) | Diff |
|---|---:|---|---:|---:|---:|
| balanced | 128 | prefill / decode | 313.102 / 21.996 ms | 313.115 / 22.193 | +0.00 % / +0.90 % |
| balanced | 512 | prefill / decode | 1241.269 / 22.167 | 1241.282 / 22.270 | +0.00 % / +0.46 % |
| efficient | 128 | prefill / decode | 985.139 / 22.613 | 985.152 / 22.836 | +0.00 % / +0.99 % |
| efficient | 512 | prefill / decode | 3951.660 / 22.783 | 3951.673 / 22.957 | +0.00 % / +0.76 % |

Decode residual at balanced/512 (+0.103 ms of 22.167): matmuls +0.152 ms (weight scale bytes now counted in the L1 feed), `GET_ROWS`
-0.035 ms and `CONCAT` -0.017 ms (ingress counted once); every other op agrees within 0.002 ms. Recovering the explorer this closely
before changing any equation is what makes the ablation in `EQUATIONS.md` a measurement of each change and not of a rewrite.

## 3. Graph equivalence with the explorer's capture

Our pp128 and pp512 fa-off captures and the explorer's (different llama.cpp commit, BF16 vs Q4_K_M weights) have identical
operation sequences, tensor names, output shapes and MACs for all 1441 nodes in both phases; the only differences are the
formats of 187 weight tensors (98 q4_K, 36 q5_K, 36 q8_0, 17 q6_K here; BF16 there). The explorer's shape-expression
extrapolation, fitted on 128 and 512 only, reproduces our independent pp8192 capture exactly (MACs equal; prefill shapes match
on all 1441 ops once KV capacity matches; the 18 decode differences are tensors sized by the capture's 4x-prompt KV capacity,
where the explorer pads live context to 256).

## 4. Held-out check of the accounting against measured CPU time (`cpu_validate.py`)

Per (phase, op class) a non-negative linear model `seconds = a * MACs + b * bytes + c` (or elements) is fitted on the pp128 and
pp512 fa-on traces, then used to predict every per-prompt fa-on graph (71-628 tokens), which the fit never saw. Features are the
engine's own MAC and byte counts. **It tests that the counts scale correctly with shape; it says nothing about the accelerator.**

| Prompt (tokens) | Prefill measured / predicted | Error | Decode measured / predicted | Error |
|---|---:|---:|---:|---:|
| repetitive (71) | 1613.2 / 1670.6 ms | +3.6 % | 66.6 / 72.2 ms | +8.6 % |
| multilingual (152) | 3458.4 / 3447.8 | -0.3 % | 68.7 / 72.2 | +5.2 % |
| narrative (191) | 4400.6 / 4303.4 | -2.2 % | 69.7 / 72.2 | +3.6 % |
| hardware (230) | 5267.3 / 5159.1 | -2.1 % | 62.0 / 72.2 | +16.4 % |
| dialogue (235) | 5414.4 / 5268.8 | -2.7 % | 66.6 / 72.2 | +8.4 % |
| math (264) | 5945.0 / 5938.4 | -0.1 % | 70.7 / 73.0 | +3.3 % |
| code (307) | 7212.8 / 6887.3 | -4.5 % | 78.8 / 73.0 | -7.3 % |
| json_records (628) | excluded | | 71.3 / 73.8 | +3.6 % |

Training fit: pp128 prefill +3.6 %, decode +0.4 %; pp512 prefill -0.3 %, decode -0.3 %.

- The `json_records` prefill is excluded on purpose: the profiled run split a 628-token prompt into two micro-batches
  (512 + 116) while the captured graph is a single 628-token graph, so they are not the same computation. An early version of the
  loader averaged the two pieces and produced a +97 % error; it now drops any phase that ran as more than one graph in a repetition.
- The first version fitted prefill and decode jointly and over-predicted decode by 26-48 %, because prefill's long times dominate a
  joint least-squares fit and CPU kernels differ between the GEMM-like and GEMV regimes. Fitting per phase fixed it.
- Decode timings are a single first-step measurement per prompt and vary by about +/-10 % on their own, so part of the decode
  error is measurement noise. Timings come from the instrumented build.

## 5. Tests (`python -m unittest discover -s tests`, 52 tests, about 40 s)

- Formats and config (8): block sizes agree with every captured weight's recorded bytes; validation rejects bad values; hashes.
- Graph structure and equations on small synthetic graphs with hand-computed answers (13): roots and storage kinds; KV read-after-write
  and write-after-read hazards; fusion rule; GEMM/GEMV cycle counts; tile-search optimality against brute force over its candidates;
  weight-format and direct-path effects; fused-attention MACs and causal tiling; both recurrence lowerings; memory-op accounting.
- Residency and schedule (11): worked eviction cases (farthest-next-use), oversized tensors, monotonicity in L1 capacity; schedule
  bounds ordered on random DAGs; secondary pool blocking; memory service as a floor.
- Real captures (13): MAC parity, attention cross-check, explorer reproduction, monotonicity (more bandwidth, clock, banks, DMA or a
  direct path is never slower), bound ordering, format-policy ordering, tile search never streams more weights than the fixed
  heuristic, batch scaling, infeasibility flags, determinism, host fallback, CPU held-out error thresholds.
- Sweeps (7): Pareto dominance and tie handling, cost sensitivity to banks and the direct path, sweep files carry the base config,
  invalid grid points flagged, ablation ends at the full engine, sensitivity direction, seeded Monte-Carlo.

Bugs the tests or checks found while building it: a schedule clip that made the list schedule shorter than its own critical path
(ops that occupy a second pool now block it); a residency crash on an infeasible design (now flagged, not raised); an inconsistent DMA
definition between copies and gathers; an activation-traffic undercount in the explorer's spill proxy (kept as the legacy switch).

Added 2026-10-05, with review fixes on 2026-10-06 and 2026-10-08 (82 tests in total, 3 of them skipped without local-only traces and summaries):

- Anchored fusion groups (9): the rule on synthetic graphs (parallel group, one anchor per group, no cross-layer fusion); the
  engine's loader and `groups.py` reproduce the 2026-10-02 dependency-map study's committed tables for the four 2026-09-24
  captures (same operations, same data and state-ordering edges including write-after-write, same layer, category,
  parallel-group and fusion-group for every operation); on the engine's own captures `groups` fuses a superset of `chains` and
  never adds activation traffic; each captured `FLASH_ATTN_EXT` is one attention anchor.
- Serial recurrence on the matrix arrays (1): MACs, time and the vector reservation for the whole compute interval (an
  independent vector op cannot start inside it) against the hand-computed equation, linear
  in the penalty, independent of the dedicated-unit count, never chosen by `auto`, valid for a single token.
- Matrix recurrence feasibility and traffic (2): on a recurrence-only workload with no recurrence unit and no scalar core,
  matrix units without the `gemm` capability leave the design infeasible; the matrix path's L1 state traffic equals the vector
  path's (39 re-reads and 16 writes of the state for 8 tokens), where a dedicated unit with scratch has none.
- CPU validation report (4): the Markdown report sums measured and predicted time per operation class over the held-out graphs;
  a run without its local-only traces or graphs is reported as NOT CHECKED, with the reason per workload, no result tables
  and a non-zero exit code; nothing is predicted when the training traces are missing.
- Switch register (5): `attention.kv_padding` is bracketed at `captured`; `schedule.within_op_overlap` is unbracketed with its
  alternative reported alone and as an upper bound; infeasible designs are never turned into a percentage (the reverted evaluation keeps its validity
  and errors); every registered path and value is valid; `config.conservative` leaves no optimistic switch and moves
  nothing else; the explorer-equations profile sits at the explorer setting of every switch; the switch table brackets the
  configured result.
- Default results were compared before and after these changes on pp128/pp512/pp8192 flash-attention-on, pp512 and one prompt
  flash-attention-off, and the explorer-equations profile: identical apart from the added `switches` list and the config hash
  (which now includes `recurrent.matrix_penalty`).

## 6. Not validated

- The accelerator timing equations (matrix cycles, L1/HBM service, scheduling): analytical, uncalibrated.
- Serial recurrence on the matrix arrays: the penalty is a swept placeholder, and neither array fill/drain per token nor the wait
  for one token's state update before the next is charged separately. The five state reads and two state writes per token
  through L1 are assumed (the same as the vector path); no array-local state storage is modelled.
- Anchored fusion groups: structural candidates. Nothing shows that a unit can apply a whole group in place.
- Chunked recurrence lowering: MAC counts derived here from the chunkwise structure, not compared with an implementation. It changes
  prefill by 25-29 % (`EQUATIONS.md` 2.8).
- Fused-attention timing and the `(8 + 2a)`-byte score hand-off; the static L1 partition; the fusion epilogue assumption.
- Batch > 1 (no batched capture), prefill micro-batching, any prompt length that was not captured.
- Cost and DSP proxies; FPGA fit; energy and power.
- Numerical accuracy of any quantisation; Q4_K_M does not validate a uniform INT4/group-64 format.
- Everything here is one host, one run per graph, CPU only: the CPU-vs-Metal logit check in the run's `analysis/` is empty on Windows.
