#!/bin/bash
set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables
ENV_FILE="$REPO_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

echo "Starting deployment for $DJANGO_PROJECT_NAME"

# 1. Create user and group
if ! getent group "$LINUX_GROUP" > /dev/null 2>&1; then
    echo "Creating group $LINUX_GROUP..."
    groupadd --system "$LINUX_GROUP"
else
    echo "Group $LINUX_GROUP already exists."
fi

if ! id "$LINUX_USER" &>/dev/null; then
    echo "Creating user $LINUX_USER..."
    sudo useradd --system --gid "$LINUX_GROUP" --shell /bin/bash --home "$LINUX_USER_HOME" "$LINUX_USER"
    echo "$LINUX_USER user created."
else
    echo "$LINUX_USER user already exists."
fi

if [ ! -d "$LINUX_USER_HOME" ]; then
    mkdir -p "$LINUX_USER_HOME"
fi
chown "$LINUX_USER:$LINUX_GROUP" "$LINUX_USER_HOME"

# 2. Install necessary system packages
echo "Checking and installing necessary system packages..."
apt-get update
apt-get install -y python3 python3-venv postgresql nginx

# 3. Move the repository to REPO_ROOT
echo "Moving repository to $REPO_ROOT..."
if [ "$REPO_DIR" != "$REPO_ROOT" ]; then
    mkdir -p "$(dirname "$REPO_ROOT")"
    mv "$REPO_DIR" "$REPO_ROOT"
    cd "$REPO_ROOT"
    echo "Repository moved to $REPO_ROOT"
else
    echo "Already in $REPO_ROOT, no need to move"
fi

# 4. Set proper ownership
chown -R "$LINUX_USER:$LINUX_GROUP" "$REPO_ROOT"

# 5. Set up the database
echo "Setting up the database..."
if sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1; then
    echo "Database user '$DB_USER' already exists. Skipping creation."
else
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
fi

if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    echo "Database '$DB_NAME' already exists. Skipping creation."
else
    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
fi

# 6. Set up Python virtual environment
run_as_user() {
    sudo -u "$LINUX_USER" XDG_RUNTIME_DIR=/run/user/$(id -u $LINUX_USER) bash -c "$1"
}
run_script_as_user() {
    sudo -u "$LINUX_USER" XDG_RUNTIME_DIR=/run/user/$(id -u $LINUX_USER) bash "$1"
}

echo "Setting up Python virtual environment..."
if [ ! -d "$REPO_ROOT/venv/bin" ]; then
    run_as_user "
        cd "$REPO_ROOT"
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    "
else
    echo "Virtual environment already exists. Skipping creation. Running pip install..."
    run_as_user "
        cd "$REPO_ROOT"
        source venv/bin/activate
        pip install -r requirements.txt
    "
fi

# 7. Create and place settings_django_deploy.py
echo "Creating settings_django_deploy.py..."
SETTINGS_TEMPLATE="$REPO_ROOT/deployment/templates/settings_django_deploy.py.template"
SETTINGS_DEPLOY_PATH="$REPO_ROOT/django/$DJANGO_PROJECT_NAME/settings_django_deploy.py"

envsubst < "$SETTINGS_TEMPLATE" > "$SETTINGS_DEPLOY_PATH"
chown $LINUX_USER:$LINUX_GROUP "$SETTINGS_DEPLOY_PATH"
echo "settings_deploy.py created at $SETTINGS_DEPLOY_PATH"

# 8. Set up Nginx service
echo "Setting up Nginx..."
NGINX_LOG_DIR="$REPO_ROOT/logs"
if [ ! -d "$NGINX_LOG_DIR" ]; then
    sudo -u $LINUX_USER mkdir -p $NGINX_LOG_DIR
    echo "Created Nginx log directory: $NGINX_LOG_DIR"
fi

envsubst '$REPO_ROOT $DOMAIN_NAME $DJANGO_PROJECT_NAME' < "$REPO_ROOT/deployment/templates/django-nginx.conf.template" > "/tmp/$DJANGO_PROJECT_NAME"
sudo mv "/tmp/$DJANGO_PROJECT_NAME" "/etc/nginx/sites-available/$DJANGO_PROJECT_NAME"
sudo ln -sf "/etc/nginx/sites-available/$DJANGO_PROJECT_NAME" "/etc/nginx/sites-enabled/"
sudo nginx -t && sudo systemctl restart nginx

# 9. Set up Gunicorn service
echo "Setting up Gunicorn start script..."
GUNICORN_SCRIPT_TEMPLATE="$REPO_DIR/deployment/templates/gunicorn_start.sh.template"
GUNICORN_SCRIPT_PATH="$REPO_ROOT/deployment/gunicorn_start.sh"
envsubst '$DJANGO_PROJECT_NAME $REPO_ROOT $LINUX_USER $LINUX_GROUP' < "$GUNICORN_SCRIPT_TEMPLATE" > "$GUNICORN_SCRIPT_PATH"
chmod +x "$GUNICORN_SCRIPT_PATH"
chown $LINUX_USER:$LINUX_GROUP "$GUNICORN_SCRIPT_PATH"

echo "Setting up Gunicorn systemd service..."
SYSTEMD_DIR="/home/$LINUX_USER/.config/systemd/user"

if [ -f "$SYSTEMD_DIR/$DJANGO_PROJECT_NAME.service" ]; then
    echo "Systemd service for $DJANGO_PROJECT_NAME already exists. Stopping service."
    run_as_user "systemctl --user stop $DJANGO_PROJECT_NAME.service"
else
    sudo -u $LINUX_USER mkdir -p $SYSTEMD_DIR
fi

envsubst < "$REPO_ROOT/deployment/templates/django-gunicorn.service.template" > "$SYSTEMD_DIR/$DJANGO_PROJECT_NAME.service"

sudo loginctl enable-linger $LINUX_USER
run_as_user "systemctl --user daemon-reload"
run_as_user "systemctl --user enable $DJANGO_PROJECT_NAME.service"
run_as_user "systemctl --user start $DJANGO_PROJECT_NAME.service"

# 10. Set up Django environment and attempt to load db.json
echo "Setting up Django environment..."
run_script_as_user "$REPO_ROOT/deployment/update.sh"

echo "Deployment completed successfully!"
echo "To check gunicorn status (as root), run: sudo -u $LINUX_USER XDG_RUNTIME_DIR=/run/user/$(id -u $LINUX_USER) systemctl --user status $DJANGO_PROJECT_NAME.service"
echo "To check Nginx status (as root), run: sudo systemctl status nginx"
