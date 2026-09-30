#!/bin/bash
# Run every 30 s by heedless-sync.timer, as the site user. When GitHub's main has moved:
# deploy it to the site (fast-forward, migrate, static files, graceful reload), then update
# the open aggregate PR using normal commits. Otherwise do nothing.
set -euo pipefail
SITE="${SITE:-/home/django/heedless-backbones}"
SETTINGS="${SETTINGS:-heedless-backbones.settings_django_deploy}"
SERVICE="${SERVICE:-heedless-backbones.service}"
STATE="$SITE/ingestion_data/main.sha"
# The deploy settings (e.g. STATIC_ROOT from REPO_ROOT) need the deploy env, expanded by bash.
set -a
# shellcheck source=/dev/null
source "$SITE/django-deploy/.env"
set +a

exec 9>"$STATE.lock"
flock -n 9 || exit 0  # the previous tick is still working

remote=$(git -C "$SITE" ls-remote origin refs/heads/main | cut -f1)
[ -n "$remote" ] || { echo "Could not read origin/main"; exit 1; }
[ "$remote" = "$(cat "$STATE" 2>/dev/null)" ] && exit 0
echo "main is now $remote"

cd "$SITE"
before=$(git rev-parse HEAD)
git fetch -q origin main
git merge -q --ff-only FETCH_HEAD  # refuses (and retries next tick) if the checkout has diverged
cd django
if git diff --name-only "$before" HEAD | grep -qx requirements.txt; then
  ../venv/bin/pip install -q -r ../requirements.txt
fi
../venv/bin/python manage.py migrate --noinput --settings="$SETTINGS"
../venv/bin/python manage.py collectstatic --noinput -v 0 --settings="$SETTINGS"
systemctl --user kill -s HUP "$SERVICE"  # gunicorn reloads its workers without dropping requests
../venv/bin/python manage.py refresh_auto_prs --settings="$SETTINGS"
echo "$remote" > "$STATE"
