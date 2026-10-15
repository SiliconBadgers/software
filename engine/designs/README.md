# Baselines and open decisions

[`characterized-v1`](baselines/characterized-v1/record.json) records source
`0e06023`, the explicit Qwen mixed-Q4_K_M capture run and the complete resolved
balanced configuration. It is a provisional reproduction baseline, not approved
hardware. [`decisions.json`](decisions.json) links compute, local-memory and joint
questions to configuration paths and evidence.

From `engine/`:

```text
python -B -m sbengine eval --run ../results/2026-09-27_1839_windows-2 --graph pp512-fa-on --profile designs/baselines/characterized-v1/profile.json
```

Resolved profiles do not merge evolving defaults. An incomplete resolved profile
fails rather than silently gaining new assumptions. Partial profiles retain their
existing override behavior.

For a reviewed revision, add a baseline directory with its full profile and a
record referencing the parent, selected study/configuration, source/capture
identities, review evidence and rationale. Preserve earlier revisions.
See [scientific stages](../../docs/DESIGN-RULES.md) for calculation ownership.
