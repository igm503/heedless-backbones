#!/bin/bash
set -e

get_config() {
    local section=$1
    local key=$2
    local file="../config.ini"

    value=$(awk -F '=' -v s="[$section]" -v k="$key" '
        $0 == s {f=1; next}
        /^\[/ {f=0}
        f && $1 ~ "^"k"[[:space:]]*" {sub(/^[^=]+=/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print; exit}
    ' "$file")

    echo "$value"
}

echo "Starting Heedless Backbones setup..."

# 1. Update configuration files
echo "Updating configuration files..."
python3 update_paths.py

# 2. Set up the database
echo "Setting up the database..."
DB_NAME=$(get_config "Database" "name")
DB_USER=$(get_config "Database" "user")
DB_PASS=$(get_config "Database" "password")
DB_HOST=$(get_config "Database" "host")
DB_PORT=$(get_config "Database" "port")

# Check if user exists
if sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1; then
    echo "Warning: Database user '$DB_USER' already exists. Skipping user creation."
else
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
fi

# Check if database exists
if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    echo "Warning: Database '$DB_NAME' already exists. Skipping database creation."
else
    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
fi

# 3. Set up Django environment
echo "Setting up Django environment..."
PROJECT_ROOT=$(get_config "Paths" "project_root")
CONDA_ENV=$(get_config "Conda" "environment")
LINUX_USER=$(get_config "User" "linux_user")
LINUX_GROUP=$(get_config "User" "linux_group")

# Function to run commands as the specified user
run_as_user() {
    sudo -H -u "$LINUX_USER" bash -c "$1"
}

# Activate Conda and run Django commands as the specified user
run_as_user "
    source ~/.bashrc
    export PATH=\"/home/$LINUX_USER/anaconda3/bin:\$PATH\"
    
    if ! command -v conda &> /dev/null; then
        echo \"Error: conda command not found. Please ensure Conda is installed and in the PATH.\"
        exit 1
    fi
    
    eval \"\$(conda shell.bash hook)\"
    conda activate $CONDA_ENV
    
    if ! command -v python &> /dev/null; then
        echo \"Error: python command not found. Please ensure Python is installed and in the PATH.\"
        exit 1
    fi
    
    cd $PROJECT_ROOT/django
    python manage.py migrate
    python manage.py collectstatic --noinput
"

# 4. Load the database dump
echo "Loading the database dump..."
cd $PROJECT_ROOT/deployment
if [ -f "db.sql" ]; then
    export PGPASSWORD="$DB_PASS"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "db.sql"
    unset PGPASSWORD
else
    echo "Warning: db.sql file not found. Skipping database dump loading."
fi

# 5. Set up Nginx
echo "Setting up Nginx..."

NGINX_LOG_DIR="$PROJECT_ROOT/logs
if [ ! -d "$NGINX_LOG_DIR" ]; then
    sudo -u $LINUX_USER mkdir -p $NGINX_LOG_DIR
    echo "Created Nginx log directory: $NGINX_LOG_DIR"
fi

if [ ! -f "/etc/nginx/sites-available/heedless-backbones" ]; then
    sudo mv heedless-backbones.nginxconf /etc/nginx/sites-available/heedless-backbones
    sudo ln -s /etc/nginx/sites-available/heedless-backbones /etc/nginx/sites-enabled/heedless-backbones
    sudo nginx -t && sudo systemctl restart nginx
else
    echo "Warning: Nginx configuration for heedless-backbones already exists. Skipping Nginx setup."
fi

# 6. Set up systemd service
echo "Setting up systemd service..."
SYSTEMD_DIR="/home/$LINUX_USER/.config/systemd/user"
sudo -u $LINUX_USER mkdir -p $SYSTEMD_DIR

sudo chmod +x $PROJECT_ROOT/deployment/gunicorn.sh
sudo chown $LINUX_USER:$LINUX_GROUP $PROJECT_ROOT/deployment/gunicorn.sh

if [ ! -f "$SYSTEMD_DIR/heedless-backbones.service" ]; then
    sudo -u $LINUX_USER mv $PROJECT_ROOT/deployment/heedless-backbones.service $SYSTEMD_DIR/heedless-backbones.service

    # Reload systemd daemon and enable/start service
    sudo loginctl enable-linger $LINUX_USER
    run_as_user "XDG_RUNTIME_DIR=/run/user/$(id -u $LINUX_USER) systemctl --user daemon-reload"
    run_as_user "XDG_RUNTIME_DIR=/run/user/$(id -u $LINUX_USER) systemctl --user enable heedless-backbones.service"
    run_as_user "XDG_RUNTIME_DIR=/run/user/$(id -u $LINUX_USER) systemctl --user start heedless-backbones.service"
else
    echo "Warning: systemd service for heedless-backbones already exists. Reloading and restarting service."
    run_as_user "XDG_RUNTIME_DIR=/run/user/$(id -u $LINUX_USER) systemctl --user daemon-reload"
    run_as_user "XDG_RUNTIME_DIR=/run/user/$(id -u $LINUX_USER) systemctl --user restart heedless-backbones.service"
fi

echo "Setup completed successfully!"
echo "To check gunicorn status, run: sudo -u $LINUX_USER XDG_RUNTIME_DIR=/run/user/$(id -u $LINUX_USER) systemctl --user status heedless-backbones.service"
echo "To check Nginx status, run: sudo systemctl status nginx"
