# Custom quantized workload

New model-dependent work targets Zeb's
[W4G16/A8G32/Q8.24/KV8G8 export](https://huggingface.co/ZebTaylor/Qwen3.5-2B-W4G16-A8G32-Q824-KV8G8-GGUF/tree/65ef4dd4495494fb671c2ca41276df5ffccf098c).
The [workload definition](../core/experiments/qwen35-2b-zeb-compact.json) pins
the export revision, both GGUF files, sizes and SHA-256 identities separately
from the upstream Qwen checkpoint revision.

This is currently a **verified semantic import**, not a runnable profiling or
performance baseline. `profile` still defaults to the historical Q4_K_M
experiment. Explicitly selecting the custom definition for `profile` or `doctor`
fails before building, downloading or producing a run. The existing
Q4_K_M definition, `characterized-v1`, captures and historical equations remain
unchanged. Older studies are not results for the custom model.

## Import

Install/use the locked optional reader group (Python 3.12+). Download either
artifact from the pinned revision linked above, outside tracked source. Then,
from the repository root:

```sh
uv run --locked --group model-inspection python core/runner/cli.py import-workload \
  work/models/Qwen3.5-2B-v1-W4G16-A8G32-Q824-KV8G8-graph.gguf \
  --artifact graph --out build/workloads/zeb-compact-v1
```

For the weighted file, use `--artifact model` and another new output directory.
No download or execution of embedded source is implicit. The importer reuses
GGML's maintained `gguf==0.19.0` reader rather than implementing a second binary
format parser. Before output, it checks:

- Exact artifact size/hash, custom architecture and metadata version, model
  identity, configuration and embedded graph identity.
- Every embedded runtime source against its recorded SHA-256 and the graph's
  source inventory. This verifies source text, not that the bundled environment
  is complete or executable.
- Static tensor shapes, storage widths, unsigned-byte interpretation and aliases;
  the weighted GGUF directory must match the manifest. The graph-only file has
  no tensor payload, even though its graph describes the same weight inventory.
- Declared DAG references/order, unique producers and outputs, weight references
  and arithmetic-contract references.

Output contains the original embedded graph text as `semantic-graph.json` and
an `import.json` with source/definition/implementation identities and a compact
operation inventory. Output stays under `build/workloads/`; existing directories
and escaping paths are rejected. There are no generated `prefill.json` or
`decode.json` captures. The existing GGML capture loader explicitly rejects a
semantic DAG.

`cdfg.semantic.SemanticGraph` preserves symbolic dimensions and source contracts.
The published contract covers batch-one initial prefill (`P=0`) and cached
single-token decode (`T=1`, `P>0`). The importer does not construct phase graphs,
validate cache replay or infer a target schedule. Source state-output descriptors
are retained verbatim, including their symbolic `P`; cache append sizes must be
established by a numerical/capture backend.

## Numerical and execution boundaries

| Area | Published custom export | Consequence for Software |
|---|---|---|
| Main weights | 150 affine GPTQ INT4 linears, group 16; byte logarithmic group scales, packed zero points and FP16 row bases | Not GGML Q4_K and not the public symmetric INT4/FP32-scale stack. Include metadata bytes. |
| Other weights | FP16 embedding/tied head, gates, norms and convolution | Not uniform INT4; retain the tied-weight alias. |
| Dense execution | All 187 linears use A8 QDQ, group 32; packed weights temporarily decode to FP16 for native GEMMs | Not evidence for native INT4 GEMM or hardware INT8 accumulation. |
| Recurrence | Ordered, saturating INT32 Q8.24 state/scan with range counters | Shared arithmetic reference with the public Q8.24 implementation; not validation of the estimator's chunked lowering. |
| Attention/cache | Both K and V use INT8 with FP16 scales per group 8; eager attention decodes temporary FP16 operands | 10 bits/cache element with scale overhead, not an unqualified 8-bit footprint. No persistent floating KV shadow. |
| Other operations | Native floating vector operations remain | Not the all-integer vector/attention candidates in the public numerical repository. |

The export contains 888 semantic nodes, 770 stored tensors (320 F16, 450 I8
containers interpreted as U8 where declared), 42 state inputs/outputs and
1,836,068,480 tensor bytes. U8 packs must not be interpreted as signed INT8
weight values. The compact storage average is about 4.7567 bits per packed
weight, including metadata; it does not describe the retained FP16 embedding.

The GGUF includes source text for the evaluated numerical recipe. At inspected
public [QwenQuantization revision `ef9c3d8`](https://github.com/zebulontaylor/QwenQuantization/tree/ef9c3d84a398d4c1b9c6a0dc654ed10e60012fde),
four embedded recurrence/benchmark files match byte-for-byte. Nine other source
paths, including the compact affine packer, KV8 cache and Transformers model
source, are absent. The embedded `PackedLinear` loader consumes the recorded
**safetensors** recipe and constructs the original model; it is not a GGUF loader.
The public repository and this export therefore must not be treated as one
interchangeable runnable package. No source bundle is executed or vendored here;
the publisher's project-code terms remain unresolved.

## Compatibility finding

Local inspection on 2026-10-23 verified both downloaded GGUF SHA-256 values,
their identical graphs, all 13 embedded source hashes and the weighted tensor
directory. The pinned Software harness (`llama.cpp f46bc30cb6a7f68a67e34a00061e20a4ad1eff43`)
was run against the **complete weighted file**, not just its graph-only sibling.
It exits 2 with `unknown model architecture: qwen3_5_compact_q824_kv8`. A tiny
Q4_K_M control in the same executable succeeds. Inspection of the pinned loader
confirms that standard `qwen35` is registered, but the custom architecture is
not. The custom GGUF has no standard `qwen35.*` hyperparameters or
`tokenizer.ggml.*` arrays; it embeds tokenizer JSON instead. Its source-style
tensor names and separate affine packs also differ from the loader's expected
GGML tensors. Renaming the architecture would not bridge those contracts.
The original explorer's separate
BF16 capture and dependency scheduler remain historical analytical paths, not
compatible custom numerical backends.

No compatible end-to-end GGUF execution/capture backend was established in the
examined sources. No custom-model logits, accuracy scores, prefill/decode traces
or performance curves were produced. Publisher evaluations remain publisher
evidence, not new Software measurements. Importing a semantic graph does not
justify feeding renamed operations or old captures into the current estimator.
The recorded upstream checkpoint/recipe lineage remains the publisher's
attribution: source safetensors equivalence and tokenizer parity were not
independently reproduced. Exact file identity is not numerical validation.

## Next dependencies

- **Zeb:** identify/publish the exact GGUF execution entry point and complete
  runtime dependencies, or a supported reconstruction into the existing
  safetensors numerical runtime, with matched tokenizer, prefill and cached-decode
  checks. Clarify the project-code terms before redistributing runtime sources.
- **Akanksha + Zeb:** agree the execution/lowering contract to capture. In
  particular, distinguish FP16 QDQ reference execution from the hardware's
  INT4/INT8 matrix path (including affine zero points/logarithmic scales) and
  ordered Q8.24 recurrence. The
  [hardware decisions](https://github.com/SiliconBadgers/architecture/blob/528a935f4fe0bd3a34b978cc2c51131c0b9991d5/decisions/compute-team-decisions.md)
  are not changed by this importer.
- Once numerical execution is qualified, capture new phase pairs with complete
  provenance, add a separate reviewed baseline, then rerun affected studies.
  Preserve the original baseline and results throughout.
