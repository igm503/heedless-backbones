#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <username> <password>"
    exit 1
fi

USERNAME=$1
PASSWORD=$2
HOSTNAME=localhost
DB_NAME=model_stats
SQL_FILE=model_stats.sql

if [ ! -f "$SQL_FILE" ]; then
    echo "Error: SQL file '$SQL_FILE' not found."
    exit 1
fi

export PGPASSWORD="$PASSWORD"

psql -h "$HOSTNAME" -U "$USERNAME" -d "$DB_NAME" -f "$SQL_FILE"

if [ $? -eq 0 ]; then
    echo "SQL dump loaded successfully into $DB_NAME"
else
    echo "Error occurred while loading the SQL dump"
    exit 1
fi

unset PGPASSWORD
