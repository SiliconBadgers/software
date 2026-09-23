# AI setup before editing or committing

Every contributor must activate this clone's commit guard. For AI-assisted
work, configure capture **before the first edit**, not just before the commit.
This rollout checks the Codex integration with Git AI 1.7.5. Have other agents'
capture setup reviewed before using them here.

## First setup

Use a normal macOS/Linux terminal with Git, Bash, curl and Python 3.11+:

```sh
bash scripts/setup-git-ai.sh
# Reopen the terminal and restart Codex so its hooks load.
python3 scripts/check-git-ai.py
```

Review the script first. It verifies the pinned official installer's checksum,
installs user-level Git AI and hooks for detected supported agents, and activates
this clone's tracked pre-commit guard. It can change shell PATH and agent settings
outside this repo. It refuses to replace existing repository hooks. If it finds
a conflict, preserve those hooks and ask the maintainer to compose them.

If this machine already has the supported setup, each additional clone only needs:

```sh
python3 scripts/install-repo-hooks.py
python3 scripts/check-git-ai.py
```

Cloning does not activate Git hooks. The installer configures this clone locally;
linked worktrees share its configuration. Windows users should follow the
[official Codex integration](https://usegitai.com/docs/agents/codex) and use a
compatible terminal for the repository scripts. This guard has been exercised
on macOS and its tests also run on Linux CI.

Our settings keep prompt content in local storage and disable OSS telemetry.
Git notes carry attribution metadata. No paid Git AI service is required. Do not
enable prompt sharing for these public repos. Repository-specific privacy or
capture overrides need review; the checker does not guess their precedence.

## Prove capture once after installation or a tooling change

Use a disposable repo with an initial commit. Open a **fresh Codex session**
there and have it create a small file through its normal editing tool. Inspect
`git-ai status --json`: expect the generated lines under AI additions and the
Codex tool/model breakdown. Commit, then check:

```sh
git-ai await --timeout 30
git-ai stats HEAD --json
```

The accepted AI lines and tool/model must survive the commit. The setup check
only validates configuration, a reachable daemon and readable attribution state.
It cannot prove that a particular editor captured its edits. A session that was
open before installation must be restarted. Do not reconstruct old changes as
though they were captured live.

## Each contribution

1. Before AI edits, run `python3 scripts/check-git-ai.py`. If it fails, fix setup
   before continuing. For a sandbox/daemon problem, inspect `git-ai bg status`
   from a normal terminal; do not routinely disable the sandbox.
2. Review and test the changes, stage only intended files, and inspect
   `git-ai status --json`. If known AI work has no attribution, stop and report
   the gap. Do not relabel it as human work.
3. Commit. The installed pre-commit hook blocks a missing/disabled setup or an
   unavailable service. For Codex-authored work, include this exact trailer:

   ```text
   Co-authored-by: Codex <noreply@openai.com>
   ```

   Keep the human Git author. This trailer is the visible GitHub co-author
   credit; it is separate from Git AI's line attribution. Do not add it to
   human-only work. Other agents should receive their own accurate disclosure.
4. Check `git-ai stats HEAD --json`. For a merge commit, also inspect
   `git notes --ref=ai show HEAD`: Git AI 1.7.5 returned zero aggregate stats for
   our scaffold merge commits even though their notes contained captured line
   ranges and Codex session/model metadata. Confirm the changed files and lines
   are present in the note; a zero summary alone is not capture proof or proof
   of human authorship. Investigate a missing note before publishing AI work.
   Push the branch and verify notes publication:

   ```sh
   git-ai await --timeout 30
   git-ai fetch-notes origin --json
   git push origin refs/notes/ai
   git ls-remote origin refs/notes/ai
   ```

   An up-to-date notes push is fine. If it is rejected, reconcile with
   `git-ai fetch-notes origin` and retry. Never force-push over other people's
   notes. A retained `last_error` in daemon status can describe an older sync
   failure, so the setup check reports it separately from current service health.
   Do not claim successful publication until the push succeeds.
5. Open a PR linked to the issue, with validation and any attribution gaps.

## What is enforced

`AGENTS.md` tells Codex what to do. It does not install anything. The local hook
blocks commits **after this clone is bootstrapped**, but Git can bypass local
hooks. Do not use `--no-verify` to hide a setup failure. The Git AI merge workflow
preserves notes across supported GitHub merge operations; it cannot recreate
capture that never happened. Human review is still needed for disclosure and
meaningful validation. CI does not prove that untracked lines were human-written.

Main requires one approval from the code owner, `@abhinavnandwani`. Admin bypass
is enabled as requested. Work on branches and use PRs for review.

References: [Codex hooks](https://usegitai.com/docs/agents/codex),
[Git AI merge workflow](https://usegitai.com/docs/team-usage/ci-workflows),
[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
