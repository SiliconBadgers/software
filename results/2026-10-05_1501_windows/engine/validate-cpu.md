# CPU accounting check: 2026-10-05_1501_windows

Generated from `engine/` by `python -m sbengine validate-cpu --run ../results/2026-10-05_1501_windows --report <path>` (`engine/sbengine/cpu_validate.py`). It needs the run's local-only operation traces and captured graphs.

**Status: checked.** 15 held-out graph phases were compared; 1 workload phase was not checked and is listed at the end.

Per operation class and phase, `seconds = a * work + b * bytes + c` is fitted (non-negative least squares) on the training graphs (pp128-fa-on, pp512-fa-on) and used to predict every held-out per-prompt graph. This checks that the engine's work and byte counts explain measured CPU time on this host. It does not check the accelerator timing equations, and the timings are one instrumented run per graph.

## Whole graph

| Set | Workload | Phase | Tokens | Measured, ms | Predicted, ms | Error |
|---|---|---|---:|---:|---:|---:|
| training | pp128-fa-on | decode | 128 | 30.0 | 30.0 | -0.0% |
| training | pp128-fa-on | prefill | 128 | 836.6 | 1,002.1 | +19.8% |
| training | pp512-fa-on | decode | 512 | 30.4 | 30.4 | +0.0% |
| training | pp512-fa-on | prefill | 512 | 3,848.4 | 3,802.6 | -1.2% |
| held out | prompt-code-fa-on | decode | 307 | 30.0 | 30.2 | +0.5% |
| held out | prompt-code-fa-on | prefill | 307 | 1,931.3 | 2,309.5 | +19.6% |
| held out | prompt-dialogue-fa-on | decode | 235 | 30.0 | 30.0 | -0.1% |
| held out | prompt-dialogue-fa-on | prefill | 235 | 1,557.1 | 1,778.4 | +14.2% |
| held out | prompt-hardware-fa-on | decode | 230 | 30.0 | 30.0 | -0.2% |
| held out | prompt-hardware-fa-on | prefill | 230 | 1,704.2 | 1,742.1 | +2.2% |
| held out | prompt-json_records-fa-on | decode | 628 | 30.3 | 30.4 | +0.2% |
| held out | prompt-math-fa-on | decode | 264 | 30.8 | 30.2 | -2.1% |
| held out | prompt-math-fa-on | prefill | 264 | 1,679.4 | 1,996.4 | +18.9% |
| held out | prompt-multilingual-fa-on | decode | 152 | 30.0 | 30.0 | -0.0% |
| held out | prompt-multilingual-fa-on | prefill | 152 | 978.7 | 1,176.3 | +20.2% |
| held out | prompt-narrative-fa-on | decode | 191 | 30.4 | 30.0 | -1.4% |
| held out | prompt-narrative-fa-on | prefill | 191 | 1,202.4 | 1,459.2 | +21.4% |
| held out | prompt-repetitive-fa-on | decode | 71 | 31.9 | 30.0 | -6.2% |
| held out | prompt-repetitive-fa-on | prefill | 71 | 486.3 | 588.6 | +21.0% |

Held-out prefill: mean absolute error 16.8%, max 21.4% (n = 7).
Held-out decode: mean absolute error 1.3%, max 6.2% (n = 8).

## Held-out prompts, prefill, by operation class

Summed over the 7 held-out graphs; the 8 classes with the largest difference.

| Operation class | Measured, ms | Predicted, ms | Difference, ms | Difference, % of all measured |
|---|---:|---:|---:|---:|
| `concat` | 1,026.7 | 2,461.8 | +1,435.0 | +15.0% |
| `gated_delta_net` | 329.1 | 390.9 | +61.8 | +0.6% |
| `matmul[q4_K]` | 3,425.1 | 3,383.1 | -41.9 | -0.4% |
| `rms_norm` | 90.7 | 112.7 | +22.0 | +0.2% |
| `ssm_conv` | 154.6 | 171.9 | +17.3 | +0.2% |
| `matmul[q5_K]` | 2,951.3 | 2,934.7 | -16.6 | -0.2% |
| `matmul[q6_K]` | 1,288.8 | 1,276.1 | -12.7 | -0.1% |
| `silu` | 14.6 | 23.3 | +8.7 | +0.1% |
| 13 other classes | 258.6 | 296.1 | +37.5 | +0.4% |
| **All** | **9,539.4** | **11,050.6** | **+1,511.2** | **+15.8%** |

## Held-out prompts, decode, by operation class

Summed over the 8 held-out graphs; the 8 classes with the largest difference.

| Operation class | Measured, ms | Predicted, ms | Difference, ms | Difference, % of all measured |
|---|---:|---:|---:|---:|
| `matmul[q6_K]` | 84.2 | 82.8 | -1.5 | -0.6% |
| `matmul[q4_K]` | 87.1 | 86.2 | -1.0 | -0.4% |
| `matmul[q5_K]` | 37.8 | 37.1 | -0.7 | -0.3% |
| `flash_attn` | 2.7 | 2.4 | -0.3 | -0.1% |
| `gated_delta_net` | 4.9 | 5.2 | +0.3 | +0.1% |
| `cpy` | 3.4 | 3.7 | +0.3 | +0.1% |
| `rms_norm` | 1.9 | 2.2 | +0.3 | +0.1% |
| `get_rows` | 10.3 | 10.5 | +0.2 | +0.1% |
| 13 other classes | 11.1 | 10.5 | -0.5 | -0.2% |
| **All** | **243.5** | **240.6** | **-2.9** | **-1.2%** |

## Training graph pp128-fa-on, prefill, by operation class

| Operation class | Measured, ms | Predicted, ms | Difference, ms | Difference, % of all measured |
|---|---:|---:|---:|---:|
| `concat` | 80.2 | 219.2 | +139.0 | +16.6% |
| `gated_delta_net` | 27.5 | 34.5 | +7.0 | +0.8% |
| `ssm_conv` | 10.2 | 15.2 | +5.0 | +0.6% |
| `swiglu` | 4.8 | 7.6 | +2.8 | +0.3% |
| `rms_norm` | 7.4 | 10.0 | +2.5 | +0.3% |
| `matmul[q4_K]` | 303.6 | 305.7 | +2.2 | +0.3% |
| `silu` | 1.2 | 2.4 | +1.3 | +0.2% |
| `matmul[q5_K]` | 269.4 | 270.6 | +1.2 | +0.1% |
| 13 other classes | 132.4 | 136.9 | +4.5 | +0.5% |
| **All** | **836.6** | **1,002.1** | **+165.5** | **+19.8%** |

## Training graph pp512-fa-on, prefill, by operation class

| Operation class | Measured, ms | Predicted, ms | Difference, ms | Difference, % of all measured |
|---|---:|---:|---:|---:|
| `concat` | 897.2 | 861.9 | -35.4 | -0.9% |
| `matmul[q4_K]` | 1,169.4 | 1,167.2 | -2.2 | -0.1% |
| `gated_delta_net` | 139.8 | 138.0 | -1.8 | -0.0% |
| `silu` | 8.0 | 6.8 | -1.3 | -0.0% |
| `ssm_conv` | 62.0 | 60.7 | -1.3 | -0.0% |
| `rms_norm` | 40.7 | 39.8 | -0.9 | -0.0% |
| `matmul[q6_K]` | 430.6 | 429.8 | -0.8 | -0.0% |
| `swiglu` | 31.0 | 30.3 | -0.7 | -0.0% |
| 13 other classes | 1,069.7 | 1,068.1 | -1.6 | -0.0% |
| **All** | **3,848.4** | **3,802.6** | **-45.8** | **-1.2%** |

A phase is left out when its trace is not comparable with the captured graph (for example a prompt the profiled run split into several micro-batches).

## Not checked

| Workload | Phase | Why it was not checked |
|---|---|---|
| prompt-json_records-fa-on | prefill | no comparable trace for this phase |
