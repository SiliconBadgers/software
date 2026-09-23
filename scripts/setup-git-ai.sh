#!/usr/bin/env bash
# Review before running: installs user-level Git AI and supported agent hooks.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'"
# Check for conflicting Git hooks before making user-level changes.
python3 scripts/install-repo-hooks.py
git_ai_setup_dir=$(mktemp -d)
trap 'rm -rf "$git_ai_setup_dir"' EXIT
git_ai_installer="$git_ai_setup_dir/install.sh"
curl -fsSL https://github.com/git-ai-project/git-ai/releases/download/v1.7.5/install.sh -o "$git_ai_installer"
git_ai_expected=11ece07a73940f1c069543896b58265a8374e4595878176ca987117dc5f85cc3
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
export PATH="$HOME/.git-ai/bin:$PATH"
git-ai config set prompt_storage local
git-ai config set telemetry_oss off
python3 scripts/check-git-ai.py
echo 'Restart your terminal and Codex. Complete the capture smoke test in docs/git-ai.md before AI-assisted work.'
