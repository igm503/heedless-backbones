#!/bin/bash
set -e

ENV_FILE="$REPO_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

sudo -u "$LINUX_USER" XDG_RUNTIME_DIR=/run/user/$(id -u $LINUX_USER) systemctl --user start $DJANGO_PROJECT_NAME.service
sudo systemctl restart nginx
