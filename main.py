#!/usr/bin/env python3
"""
Deployment entrypoint.

Serves the production API (app:app) unchanged AND mounts the multi-PDF
segregator demo at /segregator — without modifying app.py. The Dockerfile's
`COPY *.py ./` already bundles this file into the image; point the Railway
start command (and start.sh / railway.toml) at `main:app`.

    uvicorn main:app --host 0.0.0.0 --port $PORT

Routes:
    /                 → existing app.py UI/endpoints (/v1/parse, /v1/analyze, ...)
    /segregator/      → segregator UI (page count, analyze, trust-verdict panel)
    /segregator/count, /segregator/analyze
"""
from app import app
from segregator_ui import app as segregator_app

# Isolated under /segregator so it can't collide with the production API.
app.mount("/segregator", segregator_app)
