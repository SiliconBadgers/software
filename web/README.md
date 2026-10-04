# Software workspace

The [software workspace](https://siliconbadgers.com/software/) brings
model controls, captured graphs, execution estimates, studies and source links
into one application. The interface uses Zeb Taylor's preserved capture and model
([PR #6](https://github.com/SiliconBadgers/software/pull/6)), Eric Wang's resource
scheduler ([PR #9](https://github.com/SiliconBadgers/software/pull/9)) and Adrian
Luo's dependency tables ([PR #10](https://github.com/SiliconBadgers/software/pull/10)).
Raghav Iyer's expanded captures and Python engine
([PR #7](https://github.com/SiliconBadgers/software/pull/7)) run through a second
adapter in the same interface. The Studies and Sources tabs link each contribution
to its original source.

## Build locally

Prerequisites: Python 3.12 or newer and Node.js 22 or newer. No model download
is required. The Python component loads a pinned
[Pyodide runtime](https://pyodide.org/en/stable/usage/webworker.html) from jsDelivr
on first use. Its calculations run in a Web Worker; no inference server or paid
service is needed. An internet connection is required for that first runtime load.
GitHub CLI access is optional; it supplies current
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
Only declared model assets, the four BF16 capture editions and 22 mixed-format
workloads are published; weights, binaries, local traces and integration build output are excluded.

For functional browser checks, with that server running:

```sh
npm install --prefix build/browser-tests --no-save --ignore-scripts playwright@1.62.1
build/browser-tests/node_modules/.bin/playwright install chromium
NODE_PATH="$PWD/build/browser-tests/node_modules" \
  SB_WORKSPACE_URL=http://localhost:8080/ node tests/web/smoke.cjs
NODE_PATH="$PWD/build/browser-tests/node_modules" \
  SB_WORKSPACE_URL=http://localhost:8080/ node tests/web/python.cjs
```

The checks exercise navigation, hardware edits, comparisons, capture inspection,
timeline exports, invalid inputs and mobile layout. CI runs them before publication.

## Publish a branch preview

1. Create a branch from `main` and push the changes.
2. Open **Actions → Publish software workspace → Run workflow**.
3. Keep the workflow itself on **main** and enter the source branch in
   **preview_ref**. Run the workflow.
4. Open **Previews** in the workspace and refresh the list, or use the link
   in the workflow's summary. Its URL is
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

Every publication rebuilds `editions.json` from the saved editions' own branch
names and source commits. The Previews tab reads this directory from the main
site, including when opened inside a branch preview. A listed preview is a
published snapshot; its source branch may have since been merged or deleted.
Local builds list only their own edition.

Study cards open their associated engine or saved experiment. The dependency-map
card selects Adrian's original 512-token BF16 map; the expanded captures remain
available in the Graphs selector. Each browser remembers its selected engine and
that engine's last valid configuration. Browser storage is optional and no account
is required. Studies, Previews and Sources use the full content width; model
controls are available in Model, Graphs and Timeline.

## Add a study component

Add an adapter under `web/components/<study>/adapter.js` and register it in
`web/registry.json`. Keep existing experiment sources in their dated locations.
The registry records the study ID, title, contributors, source directory, original
PR, adapter module and assets to publish. Asset paths must stay inside the
repository; model weights and binary formats are rejected. Every published source
asset gets a SHA-256 entry in `source-manifest.json`.

An adapter exports `createStudy()`, which may return a study or a Promise resolving
to one. The study supplies:

| Member | Contract |
| --- | --- |
| `controls` | Entries with `key`, `label`, `group` and `type`. Number inputs also need `min` and `step`; select inputs need `[value, label]` options. |
| `defaults()` | Complete configuration object. |
| `validate(config)` | Array of configuration error messages. |
| `evaluate(config)` | Report or Promise resolving to a report with `valid`, `errors`, `resource`, `prefill` and `decode`, following the current Qwen report shape. |
| `graph` | Optional engine input inventory; the shared graph inspector loads its own registered capture assets. |
| `layerMap()` | Map from tensor ID to layer number; `-1` is embedding, `24` is output head for the current 24-layer workload. |
| `scope` | Reader-facing statement of measured inputs and modeling assumptions. |

Each phase report supplies operation rows and summaries used by the cost table,
comparison chart and execution view. See `components/qwen/adapter.js` and the
preserved `explorer/engine.js` for the complete first component. The graph controls support the original BF16 editions
and 22 mixed-format capture pairs. A different graph schema or model topology
needs an explicit extension to the shared graph view.

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

## Python component

The builder publishes an explicit list of the original `sbengine` modules and
compressed captured graphs. The adapter runs those Python files through Pyodide
0.29.5 in a worker, then translates their reports into the shared tables, comparison
charts and timeline. It does not reimplement the cost equations in JavaScript.
Only the selected capture pair is loaded into the worker. The runtime and worker
remain available for subsequent comparisons. Full parameters can be exported and
edited as dotted-path JSON keys, such as `matrix.count` or `memory.weight_path`.

The Graphs tab includes the original SVG, tensor inventory and active dependencies
for every published workload. Dependencies for the expanded captures come from
Raghav's `Graph.deps()` method, including persistent-state ordering. These views
show individual operations. Adrian's fusion grouping is available for the original
BF16 maps where those tables were produced.

The Python scheduler exposes its existing compute-pool intervals through optional
detail fields. This does not change its timing calculation: SRAM and external
memory still impose aggregate service floors. Its timeline has compute lanes,
whereas Eric's JS scheduler explicitly reserves memory intervals. An aggregate
memory bound can extend the Python result beyond its final compute bar.

The engines use different captured formats, topologies, defaults and cost models.
Their default results are not a controlled hardware comparison. Choose a workload
and hold its assumptions fixed when comparing designs within an engine.

Browser checks compare Python worker results with native Python evaluation of the
same captured workload and configuration, including an alternate weight path,
a longer capture, exports and compute-pool intervals. The worker needs no NumPy;
NumPy is used only by the separate native CPU validation tests.

The Studies tab also exposes Eric's 1,440-design recurrence sensitivity study.
Filters keep its workload, format, memory profile and matrix count fixed while
comparing vector fallback, matrix penalties and dedicated recurrence counts.
**Load design** transfers the study's complete configuration to the JS engine;
it does not apply the setup panel's current values to the saved results.
