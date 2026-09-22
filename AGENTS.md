# Working in Software

Read README.md and the relevant experiment's reproduction guide before editing.

- For AI-assisted changes, check `python3 scripts/check-git-ai.py` before editing.
  Git AI requires an installed client and enabled Codex hooks; this file alone
  does not enable attribution. Follow `docs/git-ai.md` if setup is missing.
- Preserve attribution through commits and push the associated Git AI notes.
  Report a capture failure honestly. Never reconstruct historical attribution
  as though it was captured live or label imported artifacts as new AI output.
- Preserve recorded experiment results. Write reruns to a separate directory;
  record source/model hashes, backend, host, settings and numerical checks.
- Distinguish measured CPU/Metal behavior from proposed accelerator behavior.
  Operation shares, logical bytes and MAC counts are not hardware area,
  bandwidth measurements or model-quality evaluations.
- Keep models, build trees, virtual environments and new large raw traces out
  of Git. Retain compact reproducible evidence and document external inputs.
- Run checks appropriate to changed code; the MAC example's tests do not
  validate llama.cpp or the full accelerator.
- Software owns profiling, numerical references, operation mapping and
  backend/runtime/host integration. Do not assume a separate compiler team,
  custom CPU core or custom ISA.
