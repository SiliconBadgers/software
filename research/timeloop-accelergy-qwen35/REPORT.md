# First profiled Qwen matrix mappings (2026-10-12)

Source: committed `graphs/pp128/{decode,prefill}.ops.csv` under
`experiments/llama-cpp/2026-09-24-qwen35-2b`. Layer-0 `ffn_gate-0` is node
63; `ffn_up-0` (node 64) has the same shape; `ffn_out-0` is node 66.
`result_output` is node 1440 in both graphs. The CSV provides F32 output
shapes and MAC counts. K is inferred as `MACs / (M*N)`.

| Operation | M | K | N | MACs |
|---|---:|---:|---:|---:|
| `ffn_gate/up` decode | 6,144 | 2,048 | 1 | 12,582,912 |
| `ffn_gate/up` pp128 prefill | 6,144 | 2,048 | 128 | 1,610,612,736 |
| `ffn_out` decode | 2,048 | 6,144 | 1 | 12,582,912 |
| `ffn_out` pp128 prefill | 2,048 | 6,144 | 128 | 1,610,612,736 |
| `result_output` (both phases) | 248,320 | 2,048 | 1 | 508,559,360 |

## Model results

All mappings use the same provisional 8 by 8 architecture. The two prefill
choices differ only in the N loop's placement: the first puts 16 groups outside
the GlobalBuffer; `weight-reuse` puts those groups inside it. The latter uses
1,310,720 of 2,097,152 available modeled buffer words at peak. The A-stationary
label refers to the spatial M and K distribution; it is not a cycle-accurate
model of A shadow loads or B propagation in the RTL.

| Case | Cycles | A reads from MainMemory (8-bit words) | B reads (8-bit words) | C updates (model words) |
|---|---:|---:|---:|---:|
| gate decode | 196,608 | 12,582,912 | 2,048 | 12,288 |
| gate prefill, first tiling | 25,165,824 | 201,326,592 | 262,144 | 1,572,864 |
| gate prefill, weight reuse | 25,165,824 | 12,582,912 | 262,144 | 1,572,864 |
| out decode | 196,608 | 12,582,912 | 6,144 | 12,288 |
| out prefill, first tiling | 25,165,824 | 201,326,592 | 786,432 | 1,572,864 |
| out prefill, weight reuse | 25,165,824 | 12,582,912 | 786,432 | 1,572,864 |
| output head | 7,946,240 | 508,559,360 | 2,048 | 496,640 |

Timeloop reports 100% utilization and `bandwidth throttling: 1.00` in every
case, because no finite bandwidth is specified. Each cycle count equals
`M*K*N/64`, an ideal MAC lower bound with no tile load, pipeline fill/drain,
command, or memory wait time. For gate and out prefill, moving N inside the
GlobalBuffer cuts modeled A reads 16-fold (192 MiB to 12 MiB, saving 180 MiB
per isolated operation) without changing that ideal cycle count. This is a
mapping-specific traffic result, not a claim that the BPU has the buffer or
bandwidth to achieve it.

Minimum distinct A and B sizes for gate decode are 12,582,912 and 2,048
elements; for prefill they are 12,582,912 and 262,144. All A/B read counts
meet or exceed those lower bounds. The output head requires 508,559,360
distinct 8-bit A elements (485 MiB) for one modeled output column. At the
tutorial's nominal 1 GHz, sustaining the reported ideal cycles for any decode
case with one weight read per MAC would require about 64 GB/s of weight feed.
Actual memory bandwidth and latency have not been established.

The extra C updates arise from K tiling: 2 K groups for `ffn_gate` and the
output head, 6 for `ffn_out`. Timeloop counts C as 8-bit model words, while the
RTL output and accumulators are 32-bit. The C traffic and Accelergy energy
therefore need a mixed-width model before quantitative hardware conclusions.
The chosen RTL also has no stationary-output retention control. We did not
model the proposed output-stationary mode; an earlier proposed YAML failed
factor legality and was removed.
Since all modeled K dimensions exceed 8, combining partial outputs across
K tiles also requires hardware or software outside the selected matrix RTL.
The Timeloop reduction and its ideal cycle count do not establish that path.

## Checks and next hardware inputs

`python summarize_timeloop.py` checks all seven profiled reports against
`M*K*N`, minimum tensor sizes, modeled buffer capacity, and the 64-MAC ideal
cycle count. The prior 2048-square reports had a one-axis tensor projection
that incorrectly reduced the A tensor to thousands of elements; those outputs
are archived as superseded and are excluded here.

To turn the ideal cycles into BPU timing, request from the compute/memory team:
matrix array count and committed dimensions; output retention and partial-sum
spill rules; actual PE storage; SRAM capacity, bank/port layout and sustained
bandwidth; DRAM/HBM sustained bandwidth and latency; DMA setup and overlap;
matrix/vector scheduling and command latency; and target clock. With those
inputs, the traffic tables can be compared with the explorer on identical
graph nodes and used to classify memory-bound versus compute-bound cases.
