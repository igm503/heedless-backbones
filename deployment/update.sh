#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

ENV_FILE="$REPO_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

cd "$REPO_ROOT/django"
source ../venv/bin/activate

echo "Attempting to collect static files..."
python manage.py collectstatic --noinput

echo "Attempting to apply database migrations..."
python manage.py migrate

echo "Attempting to load db.json into database..."
if [ -f "$REPO_ROOT/deployment/db.json" ]; then
    python manage.py loaddata ../deployment/db.json
else
    echo "db.json not found. Skipping loading."
fi

echo "Restarting gunicorn service..."
systemctl --user restart $DJANGO_PROJECT_NAME.service
