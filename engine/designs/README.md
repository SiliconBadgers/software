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

Study reproduction checks the full parsed JSON of both loaded phases before trace
use or evaluation, including fields the analytical model does not currently use.
`decoded_capture_sha256` records SHA-256 of UTF-8 JSON serialized with sorted object
keys, compact separators, ASCII escapes and no non-finite numbers. Whitespace,
object key order and gzip compression do not affect identity; array order and all
parsed fields do. Matching plain-only, gzip-only and both-present pairs work.
These are input-provenance checks, not a freeze on subsequent in-memory graph edits.

The original `captures` entries retain their compressed-byte SHA-256, modeled
structure SHA-256, active-node count and tensor count, in that order. Structural
characterization remains distinct from complete decoded-content identity. Repository
checks validate the added hashes against the original compressed evidence as well
as checking whichever plain/gzip captures the loader selects.

For a reviewed revision, add a baseline directory with its full profile and a
record referencing the parent, selected study/configuration, source/capture
identities, review evidence and rationale. Preserve earlier revisions.
See [scientific stages](../../docs/DESIGN-RULES.md) for calculation ownership.
