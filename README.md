---
title: Rocky - ML Bank Statement Parser
emoji: 🏔️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: t4-small
suggested_storage: small
pinned: false
---

# Rocky – ML Bank Statement Parser

LayoutLMv3-based bank statement parser with Claude Vision fallback.

## Endpoints

- `POST /v1/parse` — Parse a PDF (base64-encoded)
- `GET /v1/health` — Health check
- `POST /v1/reload` — Hot-reload model weights

## Environment Variables (set as Secrets)

- `ANTHROPIC_API_KEY` — For Claude Vision fallback
- `SUPABASE_URL` — For feedback logging (optional)
- `SUPABASE_SERVICE_KEY` — For feedback logging (optional)
