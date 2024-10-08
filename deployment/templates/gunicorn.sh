#!/bin/bash

NAME="heedless-backbones-gunicorn"           # Name of the process
DJANGODIR=/path/to/heedless-backbones/src/   # Django project directory
SOCKFILE=/path/to/gunicorn.sock        # we will communicate using this unix socket
USER=linux_user                                    # the user to run as
GROUP=linux_group                             # the group to run as
NUM_WORKERS=3                                # how many worker processes should Gunicorn spawn
DJANGO_SETTINGS_MODULE=stats_site.settings   # which settings file should Django use
DJANGO_WSGI_MODULE=stats_site.wsgi           # WSGI module name

# Activate the virtual environment
cd $DJANGODIR
eval "$(conda shell.bash hook)"
conda activate model_stats
export DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE
export PYTHONPATH=$DJANGODIR:$PYTHONPATH

# Create the run directory if it doesn't exist
RUNDIR=$(dirname $SOCKFILE)
test -d $RUNDIR || mkdir -p $RUNDIR

# Start your Django Unicorn
exec gunicorn ${DJANGO_WSGI_MODULE}:application \
  --name $NAME \
  --workers $NUM_WORKERS \
  --user=$USER --group=$GROUP \
  --bind=unix:$SOCKFILE \
  --log-level=debug \
  --log-file=-
