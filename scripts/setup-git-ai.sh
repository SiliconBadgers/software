#!/usr/bin/env bash
# Run from a normal terminal. Installs global agent hooks, not repository hooks.
set -euo pipefail

git_ai_setup_dir=$(mktemp -d)
trap 'rm -rf "$git_ai_setup_dir"' EXIT
git_ai_installer="$git_ai_setup_dir/install.sh"
git_ai_expected=11ece07a73940f1c069543896b58265a8374e4595878176ca987117dc5f85cc3
curl -fsSL https://github.com/git-ai-project/git-ai/releases/download/v1.7.5/install.sh -o "$git_ai_installer"
if command -v sha256sum >/dev/null 2>&1; then
    git_ai_actual=$(sha256sum "$git_ai_installer" | awk '{print $1}')
else
    git_ai_actual=$(shasum -a 256 "$git_ai_installer" | awk '{print $1}')
fi
if [[ "$git_ai_actual" != "$git_ai_expected" ]]; then
    echo 'Git AI installer checksum mismatch; refusing to run it.' >&2
    exit 1
fi
bash "$git_ai_installer"
"$HOME/.git-ai/bin/git-ai" config set prompt_storage local
"$HOME/.git-ai/bin/git-ai" config set telemetry_oss off
echo 'Restart Codex, then run python3 scripts/check-git-ai.py from this repository.'
