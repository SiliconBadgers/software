# Software documentation

Start with [current work](START-HERE.md) and the [analysis report](../REPORT.md).

- [Scientific stages](DESIGN-RULES.md): calculation ownership and dependencies.
- [Engine](../engine/README.md): equations, configuration and CLI usage.
- [Baselines](../engine/designs/README.md) and [studies](../engine/sweeps/README.md):
  evaluate a fixed configuration and reproduce resource comparisons.
- [Study summaries](sweeps/README.md): compact results with assumptions and provenance.
- [Memory-traffic study](../research/compute-mapping/2026-10-14-memory-traffic/README.md):
  logical-byte estimates for partial sums, DeltaNet state, matrix-to-vector
  handoffs and decode SRAM feed.
- [Component tests](../tests/README.md) and
  [compatibility checks](refactor-characterization.md): verification commands.
- [Profiling next steps](profiling-next-steps.md): workload and hardware investigations.
- [HBM counter study](../research/compiler-scheduling/2026-10-25-hbm-counter-hazard/README.md):
  head-of-line cost of the command-format progress counters under HBM latency.
- [Recorded profiling](../experiments/llama-cpp/2026-09-22/README.md):
  original procedure and findings.
- [Refactor verification](verification/refactor/README.md): supporting historical
  characterization and acceptance records.

Keep maintained guides focused on architecture and reproduction. Detailed
verification records belong with their subject; PR discussion carries revision
progress and review context.
