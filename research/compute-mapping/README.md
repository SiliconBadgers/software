# research/compute-mapping

Operation/dependency maps, offload comparisons and recommendations. Link each claim to a run or source trace; separate measured time/traffic from estimates.

See [the current assignment](../../docs/START-HERE.md).

The [Qwen3.5-2B graph experiment and explorer](../../experiments/llama-cpp/2026-09-24-qwen35-2b/README.md)
provides captured operation/dependency graphs and configurable analytical
compute mappings. Its [model assumptions](../../experiments/llama-cpp/2026-09-24-qwen35-2b/explorer/MODEL.md)
document the uncalibrated timing and resource estimates. These are inputs to
research, not a validated accelerator design or a final offload recommendation.

The [one-dimensional resource-sensitivity study](2026-09-28-resource-sensitivity/REPORT.md)
freezes Zeb's default workload and identifies model-conditional compute, L1,
HBM, recurrence and DMA plateaus before dependency-aware scheduling is added.

The [dependency/resource-aware follow-up](2026-09-30-dependency-resource-sensitivity/REPORT.md)
keeps those operation costs frozen, adds a deterministic list scheduler over
source and in-place alias dependencies, and compares how the one-dimensional
knees move under explicit resource contention.
