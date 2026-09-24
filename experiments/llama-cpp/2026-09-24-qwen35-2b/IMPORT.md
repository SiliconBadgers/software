# Import provenance

Imported on 2026-09-24 from the local `Desktop/QwenProfiling` project at commit
`5c76ea8017ed49ce3eba0d7224d9c085b356cfac` ("Add Qwen graph capture and accelerator explorer").
The two previously untracked files in `configs/` are included as saved explorer
designs. The source folder and its Git history are retained at their original
location; this is a working-file import, not a merge of Git histories.

Graphs, manifests, explorer code, capture/analysis scripts, and saved designs
are imported without changes. The experiment README adds integration and
fresh-clone instructions. The parent repository owns the llama.cpp submodule
at `308883b335798865e73f8fee0d9a5fbfa13c8480`, with its official upstream URL.

The local model and logs are copied for use on this machine and remain ignored
by Git. The native binaries/libraries are rebuilt with `scripts/build.sh` at the
new location because the original libraries embedded absolute build paths.
Original CMake caches and object files are not imported. All new build outputs
remain ignored by Git. Empty `patches/` is retained locally. CSV captures retain
their original CRLF line endings, recognized by the local `.gitattributes`.
The browser explorer works without the model or native binaries.

## Attribution

The import and integration documentation used Codex with reported model ID
`gpt-6-astra`, observed in `git-ai status --json` for this session. No subagents
were used. Git AI capture was verified on the integration documentation edit.
The original project's historical generation was not captured by this import;
copy-time attribution must not be interpreted as historical authorship evidence.
This import does not reconstruct or relabel that history.

## Import validation

- Compared 57 project files, including both saved designs, byte-for-byte with
  the source; integration documentation and submodule metadata are separate.
- Verified the copied model's SHA-256 against `manifest.json`.
- `node scripts/test_model.cjs`: all 20 checks passed.
- `./scripts/build.sh`: succeeded at the new path. Dynamic-library resolution
  points to the new experiment rather than the original source folder.
- `./scripts/capture.sh 8 128 ./logs/import-smoke`: prefill and decode succeeded,
  with finite logits, acyclic dependencies, valid source IDs, and topological
  schedules. Local logs are `logs/import-build.log` and `logs/import-smoke.log`.
  This small relocation smoke test does not replace the recorded workloads or
  validate numerical equivalence or the analytical accelerator estimates.
- Checked local asset/link targets in all five HTML entrypoints.
- Parent repository `python3.12 -m unittest discover -s tests -v`: four tests
  passed. `git diff --cached --check` passed with the CSV attributes above.
- The llama.cpp working tree is clean, the 2026-09-22 baseline is unchanged,
  and model weights, logs, native binaries, and build outputs remain ignored.
