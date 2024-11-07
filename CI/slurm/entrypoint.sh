#!/bin/bash
set -e

conda activate xorbits-jobqueue
exec /usr/local/bin/docker-entrypoint.sh "$@"
