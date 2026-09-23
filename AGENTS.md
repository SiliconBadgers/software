# Working in Software profiling and runtime evidence

Use llama.cpp profiling and numerical evidence to guide compute-unit boundaries, shared resources and host/device work. Software also owns operation mapping, backend/runtime and host integration.

## Before editing or committing

- Read README.md, CONTRIBUTING.md, docs/START-HERE.md and the linked issue.
- Before any AI edit, run `python3 scripts/check-git-ai.py`. If missing, follow
  docs/git-ai.md and activate the local commit guard. Restart Codex after first
  setup and verify actual capture. AGENTS.md does not install capture or hooks.
- Do not proceed with AI edits or commits while capture/setup is broken. Report
  gaps honestly; never fabricate historical attribution or bypass the guard.
- Before committing, inspect `git-ai status --json`; after committing inspect
  `git-ai stats HEAD --json`. Include `Co-authored-by: Codex <noreply@openai.com>`
  for Codex-authored changes. Preserve the human author and publish Git AI notes
  as described in docs/git-ai.md without force-pushing them.
- Record every exact tool/model ID used in this contribution, including model
  switches and subagents. Read the session/tool metadata; never infer a model
  from the editor name or invent a backend snapshot behind an alias.
- Include one `AI-Model: tool=reported-model-id` commit trailer per model, in
  addition to the co-author credit. For this session use its actual recorded ID,
  not a copied example. Use `AI-Model: none` only for entirely human work.
- Fill the PR model table with tool, exact reported model ID, use/scope and
  evidence source. Declare uncaptured research/review or pasted output too.
  If unavailable, write `unknown` with the reason; do not hide the capture gap.
- Keep prompt storage local. Do not commit secrets, licensed collateral, model
  weights or generated build/simulation databases.
- Use a branch and PR for @abhinavnandwani's review. Keep the issue's technical
  scope intact; scaffolding and tone changes do not authorize a new design.

## Evidence and boundaries

- Link the central Architecture diagram instead of copying it. Distinguish
  accepted interfaces, proposals, assumptions, real RTL and stubs.
- Preserve dated experiment results and slide baselines. Put new work alongside
  them with revisions, commands, inputs, tool versions and limitations.
- Run checks appropriate to changed behavior. Never claim an unrun licensed-tool
  check passed or that a MAC/stub test validates the full accelerator.
- Software includes workload mapping and runtime/host integration. Do not invent
  a compiler team, custom CPU core or custom ISA.
- Keep new runs separate from experiments/llama-cpp/2026-09-22. CPU/Metal
  timing, logical bytes and MAC counts are not ASIC area/bandwidth measurements
  or model-quality evaluations. Read the experiment reproduction guide.
