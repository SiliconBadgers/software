# Contributing

Start with `README.md` and `docs/START-HERE.md`. The linked issues define the
current assignments. Divide the work within your team and record ownership in
the issue. Use existing evidence and explicit assumptions to work in parallel.

## Before editing and before the first commit

Follow [the Git AI setup and capture check](docs/git-ai.md). Activate the local
commit guard in every clone, restart Codex after installation, and verify a real
captured edit. Run `python3 scripts/check-git-ai.py` before AI-assisted work.
The guard checks setup again at commit time. Do not wait until the work is done
to configure attribution. Human-only changes do not need an AI co-author trailer.

## Branch, validate, propose

1. Accept any pending organization invite and check repository write access.
2. Create a descriptive branch from current `main`, then make the issue's change.
3. Keep editable diagram sources and link the central architecture diagram.
   Distinguish proposed interfaces, stubs, measured results and estimates.
4. Record source revisions, commands, inputs, tool versions, assumptions and
   relevant pass/fail evidence. Use `SETUP.md` for existing example checks;
   those checks only validate the example. Mark unrun licensed-tool checks as
   unrun, not passed. Keep secrets, licenses, PDKs, model weights and generated
   build/simulation databases out of Git.
5. Inspect `git-ai status --json`, commit with accurate attribution, then inspect
   `git-ai stats HEAD --json`. Codex work needs the
   `Co-authored-by: Codex <noreply@openai.com>` trailer as well as Git AI notes.
   Follow the guide to publish notes without force-pushing them.
6. Push the branch and open a PR linked to the issue for `@abhinavnandwani` to
   review. Main requires one code-owner approval; admins can bypass. Do not
   close research/scaffold issues just because folders or templates exist.

Preserve recorded experiments and slide baselines. Put new runs and proposals
in their own locations so reviewers can compare them. A change in architecture
or team scope needs an explicit proposal, not a silent documentation rewrite.
