#!/bin/sh
set -eu

mkdir -p /app/data/uploads

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend &

exec nginx -g 'daemon off;'
