# research/compute-mapping

Operation/dependency maps, offload comparisons and recommendations. Link each claim to a run or source trace; separate measured time/traffic from estimates.

See [the current assignment](../../docs/START-HERE.md).

The [Qwen3.5-2B graph experiment and explorer](../../experiments/llama-cpp/2026-09-24-qwen35-2b/README.md)
provides captured operation/dependency graphs and configurable analytical
compute mappings. Its [model assumptions](../../experiments/llama-cpp/2026-09-24-qwen35-2b/explorer/MODEL.md)
document the uncalibrated timing and resource estimates. These are inputs to
research, not a validated accelerator design or a final offload recommendation.

The [operation dependency map](2026-10-02-op-dependency-map/README.md) derives
per-layer operation chains, persistent-state ordering, parallel projection
groups and fusion candidates from those captured graphs. It is a structural
analysis with no timing or measured traffic.
