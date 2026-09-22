# Git AI and Codex

Git AI records AI line attribution in Git notes. Codex's native hooks capture
edits; AGENTS.md only tells contributors how to use the setup. Each contributor
needs a local installation. This repository does not enforce installation on
every organization member or prove that an unattributed change was human-written.

## Install once per machine

From a normal macOS or Linux terminal with Bash, curl, Git and Python 3.11+:

```sh
bash scripts/setup-git-ai.sh
# Restart Codex so the newly installed hooks load.
python3 scripts/check-git-ai.py
```

The script checks the official v1.7.5 installer's SHA256 before running it; that
installer also checks the platform binary. It modifies user-level Git/agent
configuration, installs hooks for supported agents it detects, and updates the
shell PATH. It is not limited to this repository or to Codex. Review the script
before running it. Windows users should use the
[official installation instructions](https://usegitai.com/docs/agents/codex).

Our setup selects `prompt_storage local` and disables OSS telemetry. Prompt
content stays in the local Git AI store; attribution metadata can travel with
the repository's `refs/notes/ai`. No Git AI cloud account or team service is
required. Do not put secrets into prompts or commits.

The installer enables Codex hooks and registers `PreToolUse`, `PostToolUse` and
`Stop` commands invoking `git-ai checkpoint codex --hook-input stdin`. If a
restricted agent process cannot reach the daemon, check from a normal terminal
with `git-ai bg status` and `git-ai bg start`. Do not treat a sandbox failure as
proof that hooks are working, and do not disable the sandbox as a routine fix.

## Verify capture, not just configuration

Use a disposable Git repository with an initial commit. Open a **fresh** Codex
session there and ask it to create a small file using its normal editing tool.
Before committing, inspect `git-ai status --json`. Then commit and run:

```sh
git-ai await --timeout 30
git-ai stats HEAD --json
```

Expect the generated lines to appear under `ai_additions` / `ai_accepted` and
the Codex tool/model breakdown. Configuration checks alone cannot establish this.

The initial setup was verified on September 22, 2026 with Git AI 1.7.5 and Codex
CLI 0.155.1 on macOS: a fresh Codex session generated six lines, and the commit
retained six accepted AI lines with a `codex::gpt-6-astra` breakdown. A user
LaunchAgent keeps this machine's local daemon alive across desktop tool sessions.
The current desktop session predates hook installation; new packaging edits in
that session use explicit Git AI agent checkpoints. Imported profiling artifacts
predate setup and are not retrospectively claimed as live-captured authorship.

## Commits and GitHub

Use ordinary Git commands. Check attribution after committing and ensure notes
arrive on the remote, along with the branch. For example:

```sh
git-ai await --timeout 30
git-ai stats HEAD --json
git ls-remote origin refs/notes/ai
# If automatic synchronization did not publish the notes:
git push origin refs/notes/ai
```

Do not force-push notes over another contributor's notes. Resolve a rejected
notes update before retrying. Keep the note records when rewriting commits.

The [Git AI workflow](../.github/workflows/git-ai.yaml) preserves attribution
through GitHub merges. It uses the official CLI workflow, with a pinned,
checksum-verified installer and `contents: write` for notes. Its
[first run](https://github.com/SiliconBadgers/software/actions/runs/35798778912)
passed after the initial normal merge on September 22, 2026. The initial
Software commit retains 1,000 accepted Codex lines; 16,976 imported lines remain
untracked. This verifies capture and publication, not a separate squash/rebase
test. The workflow does not create missing workstation history or reject every
untracked contribution.

References: [Codex integration](https://usegitai.com/docs/agents/codex) and
[GitHub workflow](https://usegitai.com/docs/team-usage/ci-workflows).
