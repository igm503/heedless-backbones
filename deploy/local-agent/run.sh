#!/bin/bash
# One scheduled agent run on this Mac: tunnel to the server's database, discover and screen
# papers, have Claude Code read up to $LIMIT shortlisted papers, then copy the new PDFs to the
# server so its review page can show crops. Configuration: ~/.config/heedless-agent/env
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${HEEDLESS_AGENT_ENV:-$HOME/.config/heedless-agent/env}"
# Export everything in the config (e.g. ARXIV_TROLLER_ACCOUNT) to Django.
set -a
# shellcheck source=/dev/null
source "$CONFIG"
set +a
: "${SSH_HOST:?}" "${PYTHON:?}" "${DB_NAME:?}" "${DB_USER:?}" "${DB_PASS:?}" "${INGESTION_STORAGE:?}" "${REMOTE_STORAGE:?}"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
export DB_NAME DB_USER DB_PASS INGESTION_STORAGE DB_HOST=localhost DB_PORT="${TUNNEL_PORT:-55432}"

echo "=== $(date -u +%FT%TZ) agent run"
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "$DB_PORT:localhost:5432" "$SSH_HOST" &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null || true' EXIT
for _ in $(seq 20); do nc -z localhost "$DB_PORT" 2>/dev/null && break; sleep 0.5; done

cd "$REPO/django"
# caffeinate keeps the Mac from idle-sleeping until the run finishes (closing the lid still sleeps it).
caffeinate -i "$PYTHON" manage.py ingest_papers --limit "${LIMIT:-5}" \
  --screen-limit "${SCREEN_LIMIT:-25}" ${PUBLISH:+--publish}
# The review page on the server needs the PDFs this run downloaded.
if [ -d "$INGESTION_STORAGE/papers" ]; then
  rsync -a --rsync-path="sudo -u django rsync" "$INGESTION_STORAGE/papers/" "$SSH_HOST:$REMOTE_STORAGE/papers/"
fi
