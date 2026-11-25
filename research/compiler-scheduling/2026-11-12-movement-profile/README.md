# Qwen movement-planner sweep, 2026-11-12

This profiles the captured `pp128-fa-on` Qwen matmuls for [software issue #28](https://github.com/SiliconBadgers/software/issues/28), Task E. It runs Yoon's symbolic movement planner once for each distinct `(M, N, K)` shape in prefill and decode, then weights each result by its number of instances in that phase. A second pass invoked Yoon's `python -m sbengine.codegen.movement` CLI once per shape; all 14 outputs matched the table. Generated summaries, verification records and raw per-shape CLI JSON live under ignored `build/compiler/movement-profile/2026-11-12/`.

## Main result

All seven shapes in each phase fit the planner's **assumed fully usable 8 MiB SRAM budget**, with resident B and C and two streamed A slots. There are 187 matmul instances per phase. The peak allocation for one matmul is 3,440,640 B (3.28 MiB) in prefill and 1,028,096 B (0.98 MiB) in decode. Under these assumptions, no captured matmul requires streaming B or C. This is a per-matmul allocation; multiple matmuls and reserved SRAM regions are not modeled.

| Phase | Matmul instances | MATRIX commands | TRANSFER commands | HBM to SRAM | SRAM to HBM | Peak SRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill, 128 tokens | 187 | 179,535,872 | 102,910 | 1,942,620,160 B | 293,840,896 B | 3,440,640 B |
| Decode, one token | 187 | 29,394,944 | 102,910 | 1,881,757,696 B | 3,297,280 B | 1,028,096 B |

The totals sum command and transfer counts across all 187 matmul instances. Peak SRAM is the maximum allocation for any single plan, not a sum. The output head (`248320 × 1 × 2048`) alone accounts for 7,946,240 symbolic MATRIX commands and 508,561,408 B of HBM reads in either phase under the 8/16/8 tile setting.

## One run per shape

Counts and traffic below are **per matmul instance**. Every row passed `movement_check.check()`. All 14 rows fit the assumed 8 MiB budget.

| Phase | M×N×K | Instances | SRAM B | MATRIX | TRANSFER | HBM→SRAM B | SRAM→HBM B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | 16×128×2048 | 36 | 303,104 | 4,096 | 4 | 294,912 | 8,192 |
| Prefill | 512×128×2048 | 12 | 557,056 | 131,072 | 66 | 1,310,720 | 262,144 |
| Prefill | 2048×128×2048 | 42 | 1,343,488 | 524,288 | 258 | 4,456,448 | 1,048,576 |
| Prefill | 2048×128×6144 | 24 | 1,933,312 | 1,572,864 | 258 | 13,369,344 | 1,048,576 |
| Prefill | 4096×128×2048 | 6 | 2,392,064 | 1,048,576 | 514 | 8,650,752 | 2,097,152 |
| Prefill | 6144×128×2048 | 66 | 3,440,640 | 1,572,864 | 770 | 12,845,056 | 3,145,728 |
| Prefill | 248320×1×2048 | 1 | 1,028,096 | 7,946,240 | 31,042 | 508,561,408 | 993,280 |
| Decode | 16×1×2048 | 36 | 35,328 | 512 | 4 | 34,816 | 512 |
| Decode | 512×1×2048 | 12 | 36,864 | 16,384 | 66 | 1,050,624 | 2,048 |
| Decode | 2048×1×2048 | 42 | 43,008 | 65,536 | 258 | 4,196,352 | 8,192 |
| Decode | 2048×1×6144 | 24 | 112,640 | 196,608 | 258 | 12,589,056 | 8,192 |
| Decode | 4096×1×2048 | 6 | 51,200 | 131,072 | 514 | 8,390,656 | 16,384 |
| Decode | 6144×1×2048 | 66 | 59,392 | 196,608 | 770 | 12,584,960 | 24,576 |
| Decode | 248320×1×2048 | 1 | 1,028,096 | 7,946,240 | 31,042 | 508,561,408 | 993,280 |

## Reproduction

- Planner revision: `3e5fe717abdfbc28f887bd051d62429b7fd2ae47` (`yoon/compiler-movement`).
- Captures: `results/2026-09-27_1839_windows-2/graphs/pp128-fa-on/{prefill,decode}.json.gz`. Compressed SHA-256: prefill `d5fd2eb91c831e3c4c5966f253c82604923b322f270f2477ecf8b584922261b3`; decode `fab85ef1e7207dfd4418bf6094a61740533772641e0fb0860300a071e46e2c10`.
- Target: 8 MiB SRAM, 512 B rows and full-row transfers, 8/16/8 logical tiles, 1 B A/B, 4 B C, two A slots, `inputs_released` (3j+1) reuse, unscaled mode.
- Runtime: Python 3.11.9 on Windows. This machine's `uv` 0.11.25 rejects the repository's `>=0.12.23` requirement, so the standard-library planner path ran directly with Python.

From the repository root:

```sh
python research/compiler-scheduling/2026-11-12-movement-profile/sweep.py
python research/compiler-scheduling/2026-11-12-movement-profile/verify_cli.py
python -m unittest discover -s engine/tests -p test_movement.py -q
python scripts/check_boundaries.py
```

The sweep script lowers all active `MUL_MAT` nodes in each capture, groups the resulting plans by shape, runs `MovementPlan` and the independent checker once per shape, and multiplies each row by its graph frequency. All 187 captured matmuls lowered in each phase; all 14 representative plans passed the checker. The verification script reruns the exact CLI command Yoon specified for the representative tensor ID of each shape and checks SRAM use, command counts, bytes moved, and checker status against the sweep. All 14 CLI runs matched. The existing movement suite passed 20 tests.

## Interpretation limits

- These are symbolic transfer bytes and command counts, not measured HBM bandwidth, runtime, power, or utilization. The 8 MiB value is an unconfirmed usable SRAM budget; reservations or concurrent workloads could reduce it.
- The planner assumes host-prepared INT8 A/B, unscaled 4-byte INT32 C, and one B load plus one C store per matmul. It does not model cross-matmul fusion or persistent weights, block-scaled FP32 scale tables, packed INT4 weights, other operations, or DeltaNet state traffic.
- The 8/16/8 tile setting and A/B operand roles still differ from Zeb's proposed 8×8 Matrix unit layout. The resulting MATRIX count and addresses are not executable hardware commands. Transfer counts and SRAM fit describe Yoon's current abstract placement policy.
