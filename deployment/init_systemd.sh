#!/bin/bash

if [ ! -f "heedless-backbones.service" ]; then
    echo "Error: heedless-backbones.service file not found in the current directory."
    exit 1
fi

log_file=$(grep 'StandardOutput=append:' heedless-backbones.service | cut -d':' -f2-)

if [ -z "$log_file" ]; then
    echo "Error: Could not find log file path in systemd.conf"
    exit 1
fi

log_dir=$(dirname "$log_file")
mkdir -p "$log_dir"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create log directory: $log_dir"
    exit 1
fi

if [ ! -f "$log_file" ]; then
    touch "$log_file"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create log file: $log_file"
        exit 1
    fi
    chmod 644 "$log_file"
    echo "Log file created: $log_file"
else
    echo "Log file already exists: $log_file"
fi

mkdir -p ~/.config/systemd/user/
mv heedless-backbones.service ~/.config/systemd/user/heedless-backbones.service
systemctl --user daemon-reload
systemctl --user enable heedless-backbones.service
systemctl --user start heedless-backbones.service
loginctl enable-linger $USER
systemctl --user status heedless-backbones.service

echo "The Heedless Backbones Gunicorn service has been started."
echo "Check its status with: systemctl --user status heedless-backbones.service"
