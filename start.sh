#!/usr/bin/env bash
set -e

echo "=== ML Statement Parser Service ==="
echo "Downloading model weights if needed..."
python download_model.py

echo "Starting FastAPI server on port ${PORT:-8080}..."
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
