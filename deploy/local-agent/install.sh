#!/bin/bash
# Install (or reinstall) the scheduled agent run for the current user. Undo with:
#   launchctl bootout gui/$(id -u)/com.heedlessbackbones.agent
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET="$HOME/Library/LaunchAgents/com.heedlessbackbones.agent.plist"
CONFIG="$HOME/.config/heedless-agent/env"
[ -f "$CONFIG" ] || { echo "Create ~/.config/heedless-agent/env from env.example first"; exit 1; }
AGENT_REPO="$(set -a; source "$CONFIG"; echo "${AGENT_REPO:-$HOME/.local/share/heedless-agent/repo}")"
# The agent's own checkout: a worktree of this repo at origin/main, so scheduled runs never
# depend on (or disturb) whatever you have checked out here.
git -C "$REPO" fetch -q origin main
if [ -d "$AGENT_REPO/.git" ] || [ -f "$AGENT_REPO/.git" ]; then
  git -C "$AGENT_REPO" checkout -q --detach -f origin/main
else
  mkdir -p "$(dirname "$AGENT_REPO")"
  git -C "$REPO" worktree add -q --detach "$AGENT_REPO" origin/main
fi
AGENT_REPO="$(cd "$AGENT_REPO" && pwd -P)"
sed -e "s#__REPO__#$AGENT_REPO#g" -e "s#__HOME__#$HOME#g" "$REPO/deploy/local-agent/com.heedlessbackbones.agent.plist" > "$TARGET"
launchctl bootout "gui/$(id -u)/com.heedlessbackbones.agent" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$TARGET"
echo "Installed from $AGENT_REPO; runs every 3 hours. Log: ~/Library/Logs/heedless-agent.log. Run now: launchctl kickstart gui/$(id -u)/com.heedlessbackbones.agent"
