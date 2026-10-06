# engine: analytical accelerator model driven by captured graphs

`sbengine` estimates prefill and decode time, memory traffic, footprint and a unitless resource cost for
Qwen3.5-2B on a parameterised accelerator (matrix, vector and recurrence units, shared L1, HBM). It reads the
scheduled graphs captured by `core/` directly, so shapes, dtypes, operand slots, view chains, cache and state
tensors are the measured ones, not extrapolated.

It is the successor to the 2026-09-24 explorer model
(`experiments/llama-cpp/2026-09-24-qwen35-2b/explorer/`, Zeb Taylor, llama.cpp `308883b`). That model's structure
is kept; its equations are changed where our data allow a better one, and every change can be switched back so
its effect is measured (`docs/EQUATIONS.md`). With all switches reset, this engine reproduces the explorer's
numbers on the explorer's own graphs (prefill within 0.001 %, decode within 1 %; `docs/VALIDATION.md`).

## Status and what this is not

- **Proposal / analysis tool, not an accepted interface.** Nothing here is an agreed design of the Compute, Memory
  or Control teams. Hardware parameters are **assumptions** (`docs/ASSUMPTIONS.md`); the two saved profiles copy the
  explorer's placeholder designs. See the [central architecture diagram](https://github.com/SiliconBadgers/architecture/blob/main/docs/accelerator-diagram.md)
  for the team's current boxes; this model has not been reconciled with it.
- Outputs are analytical estimates. They are not ASIC/FPGA area, power, bandwidth or timing measurements, and not a
  model-quality evaluation. The CPU-trace check validates the engine's **work and traffic accounting**, not its
  accelerator timing equations (that needs RTL or synthesis numbers).
- Captured prompt lengths only (no shape extrapolation): 128, 512, 8192 and the eight per-prompt graphs (71-628
  tokens), each with flash attention on and off. Batch > 1 is an analytical extrapolation of the batch-1 capture.
- Mixed Q4_K_M weights (as captured) do not validate a custom uniform INT4 format; `precision.weights` lets you model
  either, and says so in the result.

Default results depend on modelling switches that are off their pessimistic setting (chunked recurrence, pool scheduling, fusion
and others). `eval` names them and every result carries the full list; `python -m sbengine switches` shows what each is worth.
At pp512 flash-attention-on, balanced design, the all-pessimistic result is 49 % slower in prefill and 11 % slower in decode than
the default (`docs/EQUATIONS.md` 2.14).

## Run it

From this directory (Python 3.13, numpy; no other dependencies):

```
python -m sbengine list                                   # workloads in the newest run under results/
python -m sbengine eval --graph pp512-fa-on               # timing, bytes, bounds, binding resource
python -m sbengine eval --graph pp512-fa-on --set memory.weight_path=\"direct\" --nodes 10
python -m sbengine sweep --graph pp512-fa-on --axis matrix.count=1,2,4,8 --axis l1.banks=32,64,128,256 --out out
python -m sbengine ablation --graph pp512-fa-off          # each equation, explorer version -> this engine
python -m sbengine switches --graph pp512-fa-on           # every modelling switch, and the result without it
python -m sbengine groups --graph pp512-fa-off            # dependency-map parallel and fusion groups
python -m sbengine sensitivity --graph pp512-fa-on        # which assumptions decide the answer
python -m sbengine montecarlo --designs accel-balanced accel-efficient
python -m sbengine parity                                 # MAC accounting vs every captured summary
python -m sbengine validate-cpu                           # held-out check against measured CPU op times (--report writes per-class tables)
python -m sbengine serve                                  # interactive modelling page: http://127.0.0.1:8765
python -m unittest discover -s tests                      # 79 tests
```

`serve` opens the modelling web page (`sbengine/web/index.html`). Unlike the static run browser in
[`results/index.html`](../results/index.html), it is a live app: every control re-runs the engine through a
local JSON API, so it needs `python -m sbengine serve` running and cannot be opened as a plain file. It binds to
127.0.0.1 only.

**Data from a fresh clone.** The committed run keeps each graph as `graphs/<name>/{prefill,decode}.json.gz`
(2.5 MB for all 22 graphs); the engine reads those or the uncompressed `.json` a local capture writes. All
workloads, `eval`, `sweep`, `ablation`, `sensitivity` and `serve` work from a clone. The CPU operation traces
are not committed, so `validate-cpu` and the measured host-fallback costs need a local run (`profile`); their
tests skip without one, as does the MAC cross-check against the local-only graph summaries.

`--run results/<run>` selects a run (default: newest with a manifest and `graphs/`). `--profile` takes a file in
`profiles/` or a path; `--set path=value` overrides any config value (values are JSON). Sweep outputs go to `out/`
(git-ignored) as `<name>.csv` plus `<name>.meta.json`, which holds the full base config, its hash, the axes, the
workload and the engine version, so a CSV can be reproduced or challenged later.

## Layout

| Path | What |
|---|---|
| `sbengine/graph.py` | Load a capture: typed nodes, view-chain roots, storage kinds (weight / KV / state / input / activation), hazards |
| `sbengine/costs.py` | Per-node equations: matmul, fused attention, recurrence, conv, vector, memory ops; tile search |
| `sbengine/groups.py` | Anchored fusion groups and parallel projection groups, ported from the operation dependency map (`memory.fusion = groups`) |
| `sbengine/memory.py` | Fusion rule and activation residency simulation (plus the explorer's spill proxy, for ablation) |
| `sbengine/schedule.py` | Serial sum, dependency-aware list schedule, lower bound |
| `sbengine/model.py` | `evaluate()`: one workload under one config; feasibility checks |
| `sbengine/config.py`, `profiles/` | Defaults, validation, overrides, hashing; `accel-balanced`, `accel-efficient`, `legacy-equations` |
| `sbengine/sweep.py` | Sweeps, Pareto fronts, ablation, sensitivity, Monte-Carlo rank stability |
| `sbengine/cpu_validate.py`, `trace.py` | Held-out validation and host-fallback costs from measured CPU op traces |
| `docs/EQUATIONS.md` | Every equation, what changed from the explorer and why, with measured effect |
| `docs/ASSUMPTIONS.md` | Register of every constant: value, status, sensitivity, who can settle it |
| `docs/VALIDATION.md` | What was checked, how, results, and what is not validated |

## Provenance

Graphs: run `2026-09-27_1839_windows-2` (llama.cpp `f46bc30`, `Qwen3.5-2B-Q4_K_M.gguf` sha256 `aaf42c8b...9223`,
Windows, i7-13620H, 6 threads). The explorer's graphs (llama.cpp `308883b`, BF16) have the same operation
sequence, names and shapes for pp128/pp512 flash-attention-off; only the 187 weight tensors' formats differ.
Reference numbers for the explorer come from running its `engine.js` under Node 24.14 (origin/main `11d36ed`).

Built with Claude Code (Claude Sonnet 5.5, model ID `claude-sonnet-5-5`); report this in the PR description per `AGENTS.md`.
