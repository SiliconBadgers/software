# Baselines and open decisions

[`characterized-v1`](baselines/characterized-v1/record.json) records reviewed source
`0e06023`, the explicit Qwen mixed-Q4_K_M capture run and the complete resolved
balanced configuration. It is a provisional reproduction baseline, not approved
hardware. [`decisions.json`](decisions.json) links compute, local-memory and joint
questions using existing configuration paths and evidence.

From `engine/`, using the existing environment:

```text
python -B -m sbengine eval --run ../results/2026-09-27_1839_windows-2 --graph pp512-fa-on --profile designs/baselines/characterized-v1/profile.json
```

Resolved profiles never merge evolving defaults. An incomplete resolved profile
fails rather than silently gaining new assumptions. Ordinary historical profiles
retain their existing override behavior.

For a manually reviewed revision, add a new baseline directory with full profile
and a record referencing the old record as parent, selected study/configuration,
source/capture identities, approval evidence and rationale. Do not overwrite v1,
automatically promote a candidate or call a sampled choice globally optimal.
Follow [design rules](../../docs/DESIGN-RULES.md). No decision-management service,
physical model or new parameter system is introduced.
