#!/usr/bin/env bash
# link ~/.claude into the persistent /workspaces volume so settings, memory,
# and .remember history survive codespace rebuilds. home is wiped on rebuild,
# /workspaces is not, so the real state lives there and ~/.claude is a symlink.
set -euo pipefail

persist_dir="/workspaces/.claude-persist"
claude_home="$HOME/.claude"

mkdir -p "$persist_dir"

# first run in a container with a real (non-symlink) ~/.claude: migrate its
# contents into the persistent dir without clobbering newer persisted files,
# then replace it with the symlink
if [ -e "$claude_home" ] && [ ! -L "$claude_home" ]; then
  cp -an "$claude_home/." "$persist_dir/" 2>/dev/null || true
  rm -rf "$claude_home"
fi

ln -sfn "$persist_dir" "$claude_home"
