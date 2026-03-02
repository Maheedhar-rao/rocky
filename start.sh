#!/usr/bin/env bash
set -e

echo "=== Rocky – ML Statement Parser (HuggingFace GPU) ==="

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || true

# If weights are baked into the image, skip download.
# Otherwise, try downloading from Supabase.
if [ -f /app/models/statement_parser/model.safetensors ]; then
    echo "Model weights found in image."
else
    echo "Downloading model weights..."
    python download_model.py
fi

echo "Starting FastAPI server on port ${PORT:-7860}..."
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}" --workers 1
