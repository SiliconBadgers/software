# Baseline and modeling assumptions

The selected candidate RTL (`4926042`) has an 8 by 8 array, signed INT8
operands, 32-bit modular sums, active/shadow A registers in each PE, B columns
streamed with skew, and output columns after `Rows + Cols + 1` cycles. It has
no output backpressure. INT4 mode is reserved and unimplemented.

The compute decision document (`70ee075`) calls for stationary output
retention and a write-versus-keep control. That control is absent from the
selected RTL. The SIMD command format and RISC-V/sequencer responsibilities
are documented, but their execution timing is not modeled here.

The Timeloop architecture is a performance exploration proxy: 64 INT8 MACs,
2 MiB GlobalBuffer, 64 modeled bytes of RegisterFile per PE, and a nominal
32 GiB MainMemory. The store sizes are provisional; the RTL does not contain
these general-purpose buffers. No finite read/write bandwidths or DMA costs
are configured. `40nm` and `1e-9` seconds/cycle are copied from a tutorial and
are not Silicon Badgers technology or clock decisions.

Timeloop's A and B words are 8 bits, but this architecture also counts C as
8-bit storage words. The RTL C path is INT32. Therefore report A/B memory
reads as 8-bit modeled words and C updates as model events. Do not interpret
the C word counts as INT32 bytes or the Accelergy energy/area as BPU estimates.
Timeloop does not implement the array's 17-cycle pipeline latency, row loading,
shadow swap hold-off, carry-save psums, or no-backpressure contract here.
Every captured matmul has K greater than 8, so it needs partial outputs from
multiple A tiles combined. The selected RTL emits each 8-element dot product
but does not implement accumulation across those K tiles. The Timeloop C
read/write reductions assume an external or future combining path.

The graph captures are F32. The INT8 execution lowering and scale traffic are
not numerically validated. Results cover isolated matrix operations, not the
vector kernels, command/DMA overhead, KV state, bank conflicts, concurrency,
or full Qwen throughput.
