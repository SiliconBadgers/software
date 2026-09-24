# Working in Software profiling and runtime evidence

Use llama.cpp profiling and numerical evidence to guide compute-unit boundaries, shared resources and host/device work. Software also owns operation mapping, backend/runtime and host integration.

## Before editing or committing

- Read README.md, CONTRIBUTING.md, docs/START-HERE.md and the linked issue.
- Do not commit secrets, licensed collateral, model weights or generated
  build/simulation databases.
- Use a branch and PR for @abhinavnandwani's review. Keep the issue's technical
  scope intact; scaffolding and tone changes do not authorize a new design.

## AI assistance

If you use an AI agent, report the tool and exact model ID when available in
the PR description or change summary. State when the model is unavailable;
do not guess. No additional software, hooks or setup is required.

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
