#!/usr/bin/env bash
set -e

echo "=== Rocky – ML Statement Parser ==="

# If weights are baked into the image, skip download.
# Otherwise, try downloading from Supabase.
if [ -f /app/models/statement_parser/model.safetensors ]; then
    echo "Model weights found in image."
else
    echo "Downloading model weights..."
    python download_model.py
fi

echo "Starting FastAPI server on port ${PORT:-8080}..."
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
