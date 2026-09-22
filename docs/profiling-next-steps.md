# Extending the llama.cpp baseline

The first [profiling study](../experiments/llama-cpp/2026-09-22/REPORT.md) is complete and [reproducible](../experiments/llama-cpp/2026-09-22/README.md). The next contribution should answer a remaining question rather than repeat initial setup.

## Useful next investigations

| Question | Useful evidence | Decision it informs |
| --- | --- | --- |
| Do the observed priorities hold for intended workloads? | Representative prompts, longer continuations and relevant context/batch comparisons with pinned configurations. Investigate the anomalous 8K tracing behavior separately. | Which workload regions and operating points deserve optimization. |
| Which numerical formats are acceptable? | Reference comparisons and task-quality checks for candidate quantization, accumulation and state formats. | Hardware arithmetic and encoding choices; the mixed Q4_K_M baseline alone does not validate custom INT4. |
| Which operations should share hardware or remain adjacent? | Selected operation/intermediate-state traces, shapes, layouts, dependencies and buffer lifetimes. | Shared arithmetic, fusion and compute-unit boundaries. |
| Is an offload useful once integration is included? | Candidate kernel or backend experiments with packing, transfer, dispatch and host-fallback costs made explicit. | Which accelerator kernels and software/device contracts to implement first. |

Keep CPU/Metal timings distinct from arithmetic counts, logical tensor sizes, measured traffic and accelerator cost estimates. The output head and projections are immediate candidates for shared-arithmetic study; the current data does not mandate an engine count or permanent Compute 1/Compute 2 ownership split.

## Work in parallel

Compute can use the saved shapes now. Memory can calculate traffic and storage from those shapes. Control can model command granularity and dependencies with provisional latencies. Verification can extend the saved inputs and outputs into operation/state fixtures. Physical Design can investigate arithmetic and SRAM costs independently.

Share each useful trace, numerical result or kernel comparison as it becomes available. Reconcile incompatible assumptions with the affected teams, and freeze only the interface details needed for the next integration step. Broader profiling and accuracy work should not block unrelated design or implementation.

These are investigation directions, not personal assignments or a mandatory sequence. Software owns the operation mapping and backend/runtime/host path; a separate compiler team or custom CPU/ISA is not required.
