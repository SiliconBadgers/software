# Software workspace

The [software workspace](https://siliconbadgers.com/software/) brings
model controls, captured graphs, execution estimates, studies and source links
into one application. The interface uses Zeb Taylor's preserved capture and model
([PR #6](https://github.com/SiliconBadgers/software/pull/6)), Eric Wang's resource
scheduler ([PR #9](https://github.com/SiliconBadgers/software/pull/9)) and Adrian
Luo's dependency tables ([PR #10](https://github.com/SiliconBadgers/software/pull/10)).
Raghav Iyer's expanded profiling package and Python engine are preserved in
[PR #7](https://github.com/SiliconBadgers/software/pull/7). The Studies and Sources
tabs link each contribution to its original source.

## Build locally

Prerequisites: Python 3.12 or newer and Node.js 22 or newer. No package installation
or model download is required. GitHub CLI access is optional; it supplies current
PR status during the build.

From the repository root:

```sh
node experiments/llama-cpp/2026-09-24-qwen35-2b/scripts/test_model.cjs
python3 -m unittest discover -s tests/web -v
python3 web/build.py --ref local --preview
python3 -m http.server 8080 --directory build/site
```

Open `http://localhost:8080`. All model calculations run in the browser.
Build output is generated under `build/` and excluded from source control.
Only declared model assets and the four existing capture editions are published;
weights, binaries, local traces and integration build output are excluded.

For functional browser checks, with that server running:

```sh
npm install --prefix build/browser-tests --no-save --ignore-scripts playwright@1.62.1
build/browser-tests/node_modules/.bin/playwright install chromium
NODE_PATH="$PWD/build/browser-tests/node_modules" \
  SB_WORKSPACE_URL=http://localhost:8080/ node tests/web/smoke.cjs
```

The checks exercise navigation, hardware edits, comparisons, capture inspection,
timeline exports, invalid inputs and mobile layout. CI runs them before publication.

## Publish a branch preview

1. Create a branch from `main` and push the changes.
2. Open **Actions → Publish software workspace → Run workflow**.
3. Keep the workflow itself on **main** and enter the source branch in
   **preview_ref**. Run the workflow.
4. Open the edition linked in the workflow's summary. Its URL is
   `/software/previews/<branch-slug>-<branch-hash>/`.
5. Open a PR when the work is ready for review. Merging it updates the main
   workspace automatically.

From a terminal with repository write access:

```sh
gh workflow run software-pages.yml --repo SiliconBadgers/software \
  --ref main -f preview_ref=feature/my-study
```

The main site stays at `/software/`. Preview editions are kept in separate
directories and survive later main deployments. Re-publishing a branch replaces
only that branch's edition. Builds record the source commit and label previews.
Publication jobs serialize updates and keep generated output history on
`gh-pages`; they do not rewrite contributor commits. The job deploys the assembled
site through GitHub Pages. Preview files are public.

## Add a study component

Add an adapter under `web/components/<study>/adapter.js` and register it in
`web/registry.json`. Keep existing experiment sources in their dated locations.
The registry records the study ID, title, contributors, source directory, original
PR, adapter module and assets to publish. Asset paths must stay inside the
repository; model weights and binary formats are rejected. Every published source
asset gets a SHA-256 entry in `source-manifest.json`.

An adapter exports `createStudy()` and returns:

| Member | Contract |
| --- | --- |
| `controls` | Entries with `key`, `label`, `group` and `type`. Number inputs also need `min` and `step`; select inputs need `[value, label]` options. |
| `defaults()` | Complete configuration object. |
| `validate(config)` | Array of configuration error messages. |
| `evaluate(config)` | Synchronous report with `valid`, `errors`, `resource`, `prefill` and `decode`, following the current Qwen report shape. |
| `graph` | Tensor inventory with `tensors` and scheduled `order`, following the captured graph format. |
| `layerMap()` | Map from tensor ID to layer number; `-1` is embedding, `24` is output head for the current 24-layer workload. |
| `scope` | Reader-facing statement of measured inputs and modeling assumptions. |

Each phase report supplies operation rows and summaries used by the cost table,
comparison chart and execution view. See `components/qwen/adapter.js` and the
preserved `explorer/engine.js` for the complete first component. The initial
graph controls support Qwen's four recorded 128/512-token prefill/decode captures;
a different graph schema or model topology needs an explicit extension to the
shared graph view rather than a second standalone UI.

Add a catalog entry under `work` for a proposed study before exposing it as a
registered runtime component. A PR being merged does not by itself register its
engine. Preserve original contributors, source revisions and assumptions when
adding adapters. Source links use the exact build commit, while PR status can
refresh through GitHub's read-only public API.

## Execution and graph views

The JS component offers a serial baseline and Eric's dependency/resource
scheduler. Both use the same operation cost equations. The resource schedule
preserves data dependencies and in-place state ordering, reserves launch, compute,
SRAM and external-memory intervals, and checks for resource overcommit. It assumes
compute/memory overlap within an operation. A compute pool processes one operation
at a time, using the configured units together. The timeline exports actual modeled
starts, ends, dependencies and reserved intervals.

Adrian's committed CSV tables supply the dependency graph. The default view groups
operations into structural fusion candidates; selecting a node shows its operations,
shapes and incoming dependencies. Grouping this visualization does not implement
fusion or change the timing estimate. The original captured SVG and JSON remain
available. Each derived graph records the hashes of its source tables in the
publication manifest.

Captured topology, hardware timing assumptions and CPU measurements are distinct.
Changing the JS prompt/context settings extrapolates costs from the original
captures; it does not create a new captured graph. Resource costs remain relative
estimates, and these calculations do not validate numerical quality or an FPGA fit.
