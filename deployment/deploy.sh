#!/bin/bash

set -e

get_config() {
    awk -F "=" "/^\[$1\]/{f=1} f==1&&/^$2=/{print \$2;exit}" "../config.ini"
}

echo "Starting Heedless Backbones setup..."

# 1. Update configuration files
echo "Updating configuration files..."
python3 update_config.py

# 2. Set up the database
echo "Setting up the database..."
DB_NAME=$(get_config "Database" "name")
DB_USER=$(get_config "Database" "user")
DB_PASS=$(get_config "Database" "password")
DB_HOST=$(get_config "Database" "host")
DB_PORT=$(get_config "Database" "port")

sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

# 3. Set up Django environment
echo "Setting up Django environment..."
PROJECT_ROOT=$(get_config "Paths" "project_root")
CONDA_ENV=$(get_config "Conda" "environment")

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

cd $PROJECT_ROOT/src

python manage.py migrate
python manage.py collectstatic --noinput

# 4. Load the database dump
echo "Loading the database dump..."
cd $PROJECT_ROOT/deployment

export PGPASSWORD="$DB_PASS"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "db.sql"
unset PGPASSWORD

# 5. Set up Nginx
echo "Setting up Nginx..."
sudo mv heedless-backbones.nginxconf /etc/nginx/sites-available/heedless-backbones
sudo ln -s /etc/nginx/sites-available/heedless-backbones /etc/nginx/sites-enabled/heedless-backbones
sudo nginx -t && sudo systemctl restart nginx

# 6. Set up systemd service
echo "Setting up systemd service..."
mkdir -p ~/.config/systemd/user/
mv heedless-backbones.service ~/.config/systemd/user/heedless-backbones.service
systemctl --user daemon-reload
systemctl --user enable heedless-backbones.service
systemctl --user start heedless-backbones.service
loginctl enable-linger $USER

echo "Setup completed successfully!"
echo "To check gunicorn status: systemctl --user status heedless-backbones.service"
echo "To check Nginx status: sudo systemctl status nginx"
