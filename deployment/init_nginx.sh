#!/bin/bash

if [ "$EUID" -ne 0 ]
  then echo "Please run as root or with sudo"
  exit
fi

if [ ! -f "heedless-backbones.nginxconf" ]; then
    echo "Error: heedless-backbones.nginxconf file not found in the current directory."
    exit 1
fi

mv heedless-backbones.nginxconf /etc/nginx/sites-available/heedless-backbones

if [ $? -ne 0 ]; then
    echo "Error: Failed to move Nginx configuration file."
    exit 1
fi

echo "Nginx configuration file moved successfully."

ln -s /etc/nginx/sites-available/heedless-backbones /etc/nginx/sites-enabled/heedless-backbones

if [ $? -ne 0 ]; then
    echo "Error: Failed to create symbolic link."
    exit 1
fi

echo "Symbolic link created successfully."

# Restart Nginx
service nginx restart

# Check if Nginx restarted successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to restart Nginx."
    exit 1
fi

echo "Nginx restarted successfully."

echo "Nginx setup completed successfully!"
