#!/bin/bash
# One scheduled agent run on this Mac: tunnel to the server's database, discover and screen
# papers, have Claude Code read up to $LIMIT shortlisted papers (validation only), copy the new
# PDFs to the server for the review page, then have the server publish the clean runs and
# record them in git (one aggregate pull request). Config: ~/.config/heedless-agent/env
# Scheduled runs use their own checkout of origin/main (AGENT_REPO, set up by install.sh), which
# is updated at the start of each run; your own checkout is never touched.
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

AGENT_REPO="${AGENT_REPO:-$HOME/.local/share/heedless-agent/repo}"
if [ -z "${HB_UPDATED:-}" ] && [ "$(cd "$REPO" && pwd -P)" = "$(cd "$AGENT_REPO" 2>/dev/null && pwd -P)" ]; then
  # Move the agent's checkout to the latest main, then run the updated copy of this script
  # (git replaces files rather than rewriting them, so the running copy is unaffected).
  if git -C "$AGENT_REPO" fetch -q origin main; then
    git -C "$AGENT_REPO" checkout -q --detach -f FETCH_HEAD
  else
    echo "Could not fetch main; running $(git -C "$AGENT_REPO" rev-parse --short HEAD)"
  fi
  HB_UPDATED=1 exec /bin/bash "$AGENT_REPO/deploy/local-agent/run.sh" "$@"
fi
# Keep the Python environment in step with requirements.txt.
REQS="$REPO/requirements.txt" STAMP="$INGESTION_STORAGE/requirements.sha"
SUM=$(shasum "$REQS" | cut -d" " -f1)
if [ "$SUM" != "$(cat "$STAMP" 2>/dev/null)" ]; then
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then "$PYTHON" -m pip install -q -r "$REQS"
  else uv pip install -q --python "$PYTHON" -r "$REQS"; fi
  mkdir -p "$INGESTION_STORAGE" && echo "$SUM" > "$STAMP"
fi
export DB_NAME DB_USER DB_PASS INGESTION_STORAGE DB_HOST=localhost DB_PORT="${TUNNEL_PORT:-55432}"

echo "=== $(date -u +%FT%TZ) agent run ($(git -C "$REPO" rev-parse --short HEAD))"
# The tunnel is restarted whenever it drops (a network change, say); the run reconnects to it.
(while true; do
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
    -L "$DB_PORT:localhost:5432" "$SSH_HOST" || echo "Database tunnel dropped; reopening"
  sleep 5
done) &
TUNNEL=$!
trap 'pkill -P $TUNNEL 2>/dev/null; kill $TUNNEL 2>/dev/null || true' EXIT
for _ in $(seq 20); do nc -z localhost "$DB_PORT" 2>/dev/null && break; sleep 0.5; done

# The review page on the server needs the PDFs this run downloads; copy them as papers finish.
sync_pdfs() {
  [ -d "$INGESTION_STORAGE/papers" ] || return 0
  rsync -a --rsync-path="sudo -u django rsync" "$INGESTION_STORAGE/papers/" "$SSH_HOST:$REMOTE_STORAGE/papers/" \
    || echo "PDF copy to the server failed"
}
(while sleep 60; do sync_pdfs; done) &
SYNC=$!
trap 'pkill -P $TUNNEL 2>/dev/null; kill $TUNNEL $SYNC 2>/dev/null || true' EXIT

cd "$REPO/django"
# caffeinate keeps the Mac from idle-sleeping until the run finishes (closing the lid still sleeps it).
caffeinate -i "$PYTHON" manage.py ingest_papers --limit "${LIMIT:-5}" --screen-limit "${SCREEN_LIMIT:-25}"
kill $SYNC 2>/dev/null || true
sync_pdfs
# Publishing happens on the server, through the same function as approvals on the review page.
if [ -n "${PUBLISH:-}" ]; then
  ssh "$SSH_HOST" "cd ${REMOTE_DJANGO:-/home/django/heedless-backbones/django} && sudo -u django ../venv/bin/python \
    manage.py publish_ready --settings=${REMOTE_SETTINGS:-heedless-backbones.settings_django_deploy}"
fi
