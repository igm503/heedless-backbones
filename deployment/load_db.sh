#!/bin/bash

CONFIG_FILE="../config.ini"

get_config() {
    awk -F "=" "/^\[$1\]/{f=1} f==1&&/^$2=/{print \$2;exit}" "$CONFIG_FILE"
}

DB_NAME=$(get_config "Database" "name")
DB_USER=$(get_config "Database" "user")
DB_PASS=$(get_config "Database" "password")
DB_HOST=$(get_config "Database" "host")
DB_PORT=$(get_config "Database" "port")

SQL_FILE="db.sql"

if [ ! -f "$SQL_FILE" ]; then
    echo "Error: SQL file '$SQL_FILE' not found."
    exit 1
fi

sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"

sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

export PGPASSWORD="$DB_PASS"

psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$SQL_FILE"

if [ $? -eq 0 ]; then
    echo "SQL dump loaded successfully into $DB_NAME"
else
    echo "Error occurred while loading the SQL dump"
    unset PGPASSWORD
    exit 1
fi

unset PGPASSWORD

echo "Database setup and data loading completed successfully!"
