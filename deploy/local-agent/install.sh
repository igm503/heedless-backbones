#!/bin/bash
# Install (or reinstall) the scheduled agent run for the current user. Undo with:
#   launchctl bootout gui/$(id -u)/com.heedlessbackbones.agent
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET="$HOME/Library/LaunchAgents/com.heedlessbackbones.agent.plist"
[ -f "$HOME/.config/heedless-agent/env" ] || { echo "Create ~/.config/heedless-agent/env from env.example first"; exit 1; }
sed -e "s#__REPO__#$REPO#g" -e "s#__HOME__#$HOME#g" "$REPO/deploy/local-agent/com.heedlessbackbones.agent.plist" > "$TARGET"
launchctl bootout "gui/$(id -u)/com.heedlessbackbones.agent" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$TARGET"
echo "Installed; runs every 3 hours. Log: ~/Library/Logs/heedless-agent.log. Run now: launchctl kickstart gui/$(id -u)/com.heedlessbackbones.agent"
