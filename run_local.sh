#!/usr/bin/env bash
# Run the ML statement parser locally (CPU/MPS) for the Funders Paradise backend.
# Backend .env should point STATEMENT_PARSER_URL=http://127.0.0.1:7860/v1/parse
set -e
cd "$(dirname "$0")"
export PARSER_TRAFFIC_PCT=100   # always use the LayoutLMv3 model (no Claude fallback)
export LOG_LEVEL=WARNING
exec ./.venv311/bin/python -m uvicorn app:app --host 127.0.0.1 --port 7860 --workers 1
