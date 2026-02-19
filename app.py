#!/usr/bin/env python3
"""
FastAPI service for ML bank statement parsing.

Endpoints:
  POST /v1/parse           — Parse a PDF, returns transactions + metrics
  GET  /v1/health          — Health check + model info
  POST /v1/rollout         — Update traffic percentage
  POST /v1/reload          — Reload model weights
  GET  /v1/feedback/logs   — Query extraction feedback (Supabase)
  GET  /v1/feedback/flagged — Low-confidence extractions grouped by bank
  GET  /v1/feedback/stats  — Retraining readiness (auto-labeled pages, etc.)
  POST /v1/retrain         — Trigger background retrain with auto-labeled data
  GET  /v1/retrain/status  — Check retrain progress

Self-improving loop: when ML model has low confidence, Claude Vision fallback
simultaneously serves the user AND generates training data. Auto-labeled pages
accumulate until /v1/retrain is triggered (or MIN_PAGES_FOR_RETRAIN reached).
"""

import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from typing import Optional
from pydantic import BaseModel

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

# Config
PARSER_TRAFFIC_PCT = int(os.environ.get("PARSER_TRAFFIC_PCT", "100"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
CLAUDE_COST_PER_PAGE = float(os.environ.get("CLAUDE_COST_PER_PAGE", "0.003"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")
MIN_PAGES_FOR_RETRAIN = int(os.environ.get("MIN_PAGES_FOR_RETRAIN", "50"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("statement-parser")

app = FastAPI(title="ML Statement Parser", version="1.0.0")

# Singletons
_anthropic_client = None
_supabase_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _get_supabase():
    global _supabase_client
    if _supabase_client is None and SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# --- Request/Response models ---

class ParseRequest(BaseModel):
    pdf_base64: str
    deal_id: str = None
    shadow: bool = False  # If true, also run Claude Vision and compare

class ParseResponse(BaseModel):
    transactions: list
    total_deposits: float
    total_withdrawals: float
    average_daily_balance: Optional[float] = None
    nsf_count: int
    page_count: int
    method: str  # "layoutlmv3" or "claude_vision"
    model_version: str = ""
    low_confidence_pages: list = []
    processing_time_ms: int = 0

class RolloutRequest(BaseModel):
    traffic_pct: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    traffic_pct: int
    uptime_seconds: int

_start_time = time.time()


# --- Claude Vision fallback ---

CLAUDE_VISION_PROMPT = """Extract ALL transactions from this bank statement page. Return JSON:
{
  "transactions": [
    {"date": "01/15", "description": "DIRECT DEPOSIT", "amount": 2500.00, "type": "credit", "balance_after": 5000.00}
  ]
}
Only return valid JSON. amount is positive for credits, negative for debits."""


def _parse_with_claude_vision(pdf_bytes: bytes) -> dict:
    """Fallback: parse statement using Claude Vision."""
    from pdf2image import convert_from_bytes

    client = _get_anthropic()
    images = convert_from_bytes(pdf_bytes, dpi=200)

    all_transactions = []
    per_page_labels = []
    total_cost = 0.0

    for page_idx, image in enumerate(images):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

        response = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": CLAUDE_VISION_PROMPT},
                ],
            }],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = [l for l in lines[1:] if not l.startswith("```")]
            text = "\n".join(json_lines)

        page_txns = []
        try:
            page_data = json.loads(text)
            for t in page_data.get("transactions", []):
                t["page"] = page_idx
                t["confidence"] = 1.0
                page_txns.append(t)
                all_transactions.append(t)
        except json.JSONDecodeError:
            logger.warning(f"Claude Vision JSON parse error on page {page_idx}")

        per_page_labels.append({"transactions": page_txns, "page_index": page_idx})
        total_cost += CLAUDE_COST_PER_PAGE

    # Compute metrics
    total_deposits = sum(
        abs(t["amount"]) for t in all_transactions
        if t.get("type") == "credit" or (isinstance(t.get("amount"), (int, float)) and t["amount"] > 0)
    )
    total_withdrawals = sum(
        abs(t["amount"]) for t in all_transactions
        if t.get("type") == "debit" or (isinstance(t.get("amount"), (int, float)) and t["amount"] < 0)
    )
    balances = [t["balance_after"] for t in all_transactions if t.get("balance_after") is not None]
    nsf_count = sum(
        1 for t in all_transactions
        if any(kw in (t.get("description", "").upper()) for kw in ("NSF", "INSUFFICIENT", "OVERDRAFT"))
    )

    return {
        "transactions": all_transactions,
        "total_deposits": round(total_deposits, 2),
        "total_withdrawals": round(total_withdrawals, 2),
        "average_daily_balance": round(sum(balances) / len(balances), 2) if balances else None,
        "nsf_count": nsf_count,
        "page_count": len(images),
        "method": "claude_vision",
        "model_version": "claude-sonnet",
        "low_confidence_pages": [],
        "claude_cost": total_cost,
        "_per_page": per_page_labels,
    }


# --- Auto-labeling for retraining ---

def _save_auto_labels(pdf_bytes: bytes, per_page_labels: list, pdf_hash: str):
    """Save Claude Vision fallback results as training data for future retraining.

    When the ML model has low confidence and falls back to Claude Vision, we save
    both the Claude transaction labels AND pdfplumber word tokens in the formats
    expected by align_labels.py. This creates a self-improving loop:
      1. Low confidence → Claude Vision fallback (serves user + generates labels)
      2. Labels accumulate in data/labels/auto_{hash}/
      3. /v1/retrain aligns labels → BIO training data → retrain → promote if better
    """
    import pdfplumber as _pdfp

    stem = f"auto_{pdf_hash}"
    labels_dir = PROJECT_DIR / "data" / "labels" / stem
    pages_dir = PROJECT_DIR / "data" / "pages" / stem

    # Skip if already saved (dedup by hash)
    if labels_dir.exists():
        return

    labels_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    try:
        pdf = _pdfp.open(io.BytesIO(pdf_bytes))
    except Exception:
        return

    for page_data in per_page_labels:
        page_idx = page_data["page_index"]
        txns = page_data.get("transactions", [])

        # Save labels in label_with_claude.py format
        label_path = labels_dir / f"page_{page_idx}_labels.json"
        label_out = {
            "metadata": {},
            "transactions": txns,
            "has_transactions": bool(txns),
            "notes": "auto-labeled from Claude Vision fallback",
            "_source_image": f"page_{page_idx}.png",
            "_pdf_stem": stem,
        }
        with open(label_path, "w") as f:
            json.dump(label_out, f, indent=2)

        # Save pdfplumber words in pdf_to_pages.py format
        if page_idx < len(pdf.pages):
            page = pdf.pages[page_idx]
            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            pw, ph = page.width, page.height
            word_list = []
            for w in words:
                text = w["text"].strip()
                if not text:
                    continue
                word_list.append({
                    "text": text,
                    "bbox": [
                        int(max(0, min(1000, w["x0"] / pw * 1000))),
                        int(max(0, min(1000, w["top"] / ph * 1000))),
                        int(max(0, min(1000, w["x1"] / pw * 1000))),
                        int(max(0, min(1000, w["bottom"] / ph * 1000))),
                    ],
                    "top": round(w["top"] / ph * 1000, 2),
                    "bottom": round(w["bottom"] / ph * 1000, 2),
                })
            words_path = pages_dir / f"page_{page_idx}_words.json"
            with open(words_path, "w") as f:
                json.dump({
                    "pdf": f"auto_{pdf_hash}.pdf",
                    "page_index": page_idx,
                    "total_pages": len(pdf.pages),
                    "used_ocr": False,
                    "word_count": len(word_list),
                    "words": word_list,
                }, f, indent=2)

    pdf.close()
    logger.info(f"Auto-labeled {len(per_page_labels)} pages → data/labels/{stem}/")


# --- Feedback logging ---

def _log_feedback(deal_id: str, method: str, result: dict, shadow_result: dict = None, duration_ms: int = 0):
    """Log extraction result to statement_extraction_feedback table."""
    sb = _get_supabase()
    if not sb:
        return

    row = {
        "deal_id": deal_id,
        "method": method,
        "transaction_count": len(result.get("transactions", [])),
        "total_deposits": result.get("total_deposits", 0),
        "total_withdrawals": result.get("total_withdrawals", 0),
        "page_count": result.get("page_count", 0),
        "low_confidence_pages": result.get("low_confidence_pages", []),
        "processing_time_ms": duration_ms,
        "model_version": result.get("model_version", ""),
    }

    if shadow_result:
        row["shadow_method"] = shadow_result.get("method", "")
        row["shadow_transaction_count"] = len(shadow_result.get("transactions", []))
        row["shadow_total_deposits"] = shadow_result.get("total_deposits", 0)
        row["shadow_total_withdrawals"] = shadow_result.get("total_withdrawals", 0)

    try:
        sb.table("statement_extraction_feedback").insert(row).execute()
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")


# --- Endpoints ---

@app.post("/v1/parse", response_model=ParseResponse)
async def parse_statement(req: ParseRequest):
    """Parse a bank statement PDF."""
    start = time.time()

    try:
        pdf_bytes = base64.b64decode(req.pdf_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 PDF data")

    if len(pdf_bytes) < 100:
        raise HTTPException(400, "PDF data too small")

    # Decide: LayoutLMv3 or Claude Vision
    use_ml = random.randint(1, 100) <= PARSER_TRAFFIC_PCT
    shadow_result = None

    if use_ml:
        try:
            from predict import get_parser
            parser = get_parser()
            ml_result = parser.parse_pdf(pdf_bytes)
            result = asdict(ml_result)
            result["method"] = "layoutlmv3"

            # If low-confidence pages, fall back to Claude for those pages
            if ml_result.low_confidence_pages and len(ml_result.low_confidence_pages) > ml_result.page_count * 0.5:
                logger.info("Too many low-confidence pages, falling back to Claude Vision")
                result = _parse_with_claude_vision(pdf_bytes)

            # Shadow mode: also run Claude Vision for comparison
            if req.shadow:
                shadow_result = _parse_with_claude_vision(pdf_bytes)

        except Exception as e:
            logger.error(f"LayoutLMv3 error: {e}, falling back to Claude Vision")
            result = _parse_with_claude_vision(pdf_bytes)
    else:
        result = _parse_with_claude_vision(pdf_bytes)

    duration_ms = int((time.time() - start) * 1000)
    result["processing_time_ms"] = duration_ms

    # Log feedback
    if req.deal_id:
        _log_feedback(
            deal_id=req.deal_id,
            method=result.get("method", "unknown"),
            result=result,
            shadow_result=shadow_result,
            duration_ms=duration_ms,
        )

    # Auto-label: save Claude Vision results as training data for retraining
    per_page = result.pop("_per_page", None)
    if per_page:
        pdf_hash = hashlib.md5(pdf_bytes[:4096]).hexdigest()[:12]
        try:
            _save_auto_labels(pdf_bytes, per_page, pdf_hash)
        except Exception as e:
            logger.error(f"Auto-label save error: {e}")

    if shadow_result:
        shadow_per_page = shadow_result.pop("_per_page", None)
        if shadow_per_page:
            pdf_hash = hashlib.md5(pdf_bytes[:4096]).hexdigest()[:12]
            try:
                _save_auto_labels(pdf_bytes, shadow_per_page, f"shadow_{pdf_hash}")
            except Exception as e:
                logger.error(f"Shadow auto-label save error: {e}")

    return ParseResponse(**{k: v for k, v in result.items() if k in ParseResponse.model_fields})


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    model_loaded = False
    model_version = "none"

    try:
        from predict import get_parser
        p = get_parser()
        model_loaded = True
        model_version = p.version
    except Exception:
        pass

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        traffic_pct=PARSER_TRAFFIC_PCT,
        uptime_seconds=int(time.time() - _start_time),
    )


@app.post("/v1/rollout")
async def update_rollout(req: RolloutRequest):
    global PARSER_TRAFFIC_PCT
    old = PARSER_TRAFFIC_PCT
    PARSER_TRAFFIC_PCT = max(0, min(100, req.traffic_pct))
    logger.info(f"Traffic rollout: {old}% → {PARSER_TRAFFIC_PCT}%")
    return {"old_pct": old, "new_pct": PARSER_TRAFFIC_PCT}


@app.post("/v1/reload")
async def reload_model():
    from predict import reload_parser
    parser = reload_parser()
    logger.info(f"Model reloaded: {parser.version}")
    return {"status": "reloaded", "version": parser.version}


@app.get("/v1/feedback/logs")
async def feedback_logs(deal_id: str = None, method: str = None, limit: int = 50):
    sb = _get_supabase()
    if not sb:
        raise HTTPException(503, "Supabase not configured")

    query = sb.table("statement_extraction_feedback").select("*").order("created_at", desc=True).limit(limit)

    if deal_id:
        query = query.eq("deal_id", deal_id)
    if method:
        query = query.eq("method", method)

    result = query.execute()
    return {"logs": result.data or [], "count": len(result.data or [])}


@app.get("/v1/feedback/shadow")
async def shadow_comparison(limit: int = 50):
    """Show shadow mode comparisons (ML vs Claude Vision on same PDFs)."""
    sb = _get_supabase()
    if not sb:
        raise HTTPException(503, "Supabase not configured")

    result = (
        sb.table("statement_extraction_feedback")
        .select("*")
        .not_.is_("shadow_method", "null")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    rows = result.data or []
    comparisons = []
    for r in rows:
        ml_txns = r.get("transaction_count", 0)
        shadow_txns = r.get("shadow_transaction_count", 0)
        ml_deposits = r.get("total_deposits", 0)
        shadow_deposits = r.get("shadow_total_deposits", 0)

        comparisons.append({
            "deal_id": r.get("deal_id"),
            "method": r.get("method"),
            "ml_transactions": ml_txns,
            "claude_transactions": shadow_txns,
            "txn_diff": ml_txns - shadow_txns,
            "ml_deposits": ml_deposits,
            "claude_deposits": shadow_deposits,
            "deposit_diff": round(ml_deposits - shadow_deposits, 2),
            "processing_time_ms": r.get("processing_time_ms"),
        })

    return {"comparisons": comparisons, "count": len(comparisons)}


# --- Web UI ---

UPLOAD_HTML = """<!DOCTYPE html>
<html><head>
<title>Bank Statement Parser</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f5f5; color: #333; }
  .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 24px; margin-bottom: 8px; }
  .subtitle { color: #666; margin-bottom: 24px; }
  .upload-area { background: white; border: 2px dashed #ccc; border-radius: 12px; padding: 48px; text-align: center; cursor: pointer; transition: border-color 0.2s; }
  .upload-area:hover, .upload-area.dragover { border-color: #007aff; background: #f0f7ff; }
  .upload-area input { display: none; }
  .upload-area p { font-size: 16px; color: #666; }
  .upload-area .icon { font-size: 48px; margin-bottom: 12px; }
  .btn { background: #007aff; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 16px; cursor: pointer; }
  .btn:disabled { background: #ccc; }
  .btn:hover:not(:disabled) { background: #0056b3; }
  #results { margin-top: 24px; display: none; }
  .summary { background: white; border-radius: 12px; padding: 24px; margin-bottom: 16px; display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; }
  .stat { text-align: center; }
  .stat .value { font-size: 28px; font-weight: 700; color: #007aff; }
  .stat .label { font-size: 13px; color: #888; margin-top: 4px; }
  table { width: 100%; background: white; border-radius: 12px; overflow: hidden; border-collapse: collapse; }
  th { background: #f8f8f8; text-align: left; padding: 12px 16px; font-size: 13px; color: #666; border-bottom: 1px solid #eee; }
  td { padding: 10px 16px; border-bottom: 1px solid #f0f0f0; font-size: 14px; }
  tr:hover { background: #fafafa; }
  .credit { color: #34c759; }
  .debit { color: #ff3b30; }
  .spinner { display: none; margin: 24px auto; text-align: center; }
  .spinner.active { display: block; }
  .spinner .dot { display: inline-block; width: 10px; height: 10px; background: #007aff; border-radius: 50%; margin: 0 4px; animation: bounce 1.4s infinite ease-in-out both; }
  .spinner .dot:nth-child(1) { animation-delay: -0.32s; }
  .spinner .dot:nth-child(2) { animation-delay: -0.16s; }
  @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
  .meta { color: #999; font-size: 13px; margin-top: 12px; text-align: center; }
</style>
</head><body>
<div class="container">
  <h1>Bank Statement Parser</h1>
  <p class="subtitle">Upload a bank statement PDF to extract transactions using LayoutLMv3</p>

  <div class="upload-area" id="dropzone" onclick="document.getElementById('fileInput').click()">
    <div class="icon">&#128196;</div>
    <p id="dropLabel">Drop a PDF here or click to upload</p>
    <input type="file" id="fileInput" accept=".pdf">
  </div>

  <div class="spinner" id="spinner"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>

  <div id="results">
    <div class="summary" id="summary"></div>
    <table>
      <thead><tr><th>Date</th><th>Description</th><th>Amount</th><th>Type</th><th>Balance</th><th>Conf</th></tr></thead>
      <tbody id="txnBody"></tbody>
    </table>
    <p class="meta" id="meta"></p>
  </div>
</div>
<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const spinner = document.getElementById('spinner');
const results = document.getElementById('results');

['dragover','dragenter'].forEach(e => dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.add('dragover'); }));
['dragleave','drop'].forEach(e => dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.remove('dragover'); }));
dropzone.addEventListener('drop', ev => { if(ev.dataTransfer.files.length) { fileInput.files = ev.dataTransfer.files; handleFile(ev.dataTransfer.files[0]); }});
fileInput.addEventListener('change', () => { if(fileInput.files.length) handleFile(fileInput.files[0]); });

async function handleFile(file) {
  document.getElementById('dropLabel').textContent = file.name;
  spinner.classList.add('active');
  results.style.display = 'none';

  const reader = new FileReader();
  reader.onload = async () => {
    const b64 = reader.result.split(',')[1];
    try {
      const resp = await fetch('/v1/parse', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pdf_base64: b64})
      });
      const data = await resp.json();
      showResults(data);
    } catch(e) {
      alert('Error: ' + e.message);
    }
    spinner.classList.remove('active');
  };
  reader.readAsDataURL(file);
}

function showResults(data) {
  results.style.display = 'block';
  const s = document.getElementById('summary');
  s.innerHTML = `
    <div class="stat"><div class="value">${data.transactions?.length || 0}</div><div class="label">Transactions</div></div>
    <div class="stat"><div class="value">$${(data.total_deposits||0).toLocaleString()}</div><div class="label">Deposits</div></div>
    <div class="stat"><div class="value">$${(data.total_withdrawals||0).toLocaleString()}</div><div class="label">Withdrawals</div></div>
    <div class="stat"><div class="value">${data.page_count||0}</div><div class="label">Pages</div></div>
    <div class="stat"><div class="value">${data.nsf_count||0}</div><div class="label">NSF Fees</div></div>
  `;
  const body = document.getElementById('txnBody');
  body.innerHTML = (data.transactions||[]).map(t => `
    <tr>
      <td>${t.date||''}</td>
      <td>${(t.description||'').substring(0,50)}</td>
      <td class="${t.type}">$${(t.amount||0).toLocaleString(undefined,{minimumFractionDigits:2})}</td>
      <td>${t.type||''}</td>
      <td>${t.balance_after!=null ? '$'+t.balance_after.toLocaleString(undefined,{minimumFractionDigits:2}) : ''}</td>
      <td>${t.confidence ? (t.confidence*100).toFixed(0)+'%' : ''}</td>
    </tr>`).join('');
  document.getElementById('meta').textContent = `Method: ${data.method} | Model: ${data.model_version} | ${data.processing_time_ms}ms`;
}
</script>
</body></html>"""


# --- Spending Analysis Demo ---

@app.post("/v1/analyze")
async def analyze_statements(files: list[UploadFile] = File(...)):
    """Parse multiple PDFs and return spending analysis."""
    from spending_demo import categorize_transaction, parse_date, infer_year_from_pdf, compute_summary
    from predict import get_parser

    parser = get_parser()
    all_transactions = []

    for f in files:
        pdf_bytes = await f.read()
        result = parser.parse_pdf(pdf_bytes)
        # Try filename first, then read year from PDF content
        year_hint = None
        end_month = None
        match = re.search(r"20[2-3]\d", f.filename or "")
        if match:
            year_hint = int(match.group())
        if not year_hint:
            year_hint, end_month = infer_year_from_pdf(pdf_bytes)

        for txn in result.transactions:
            dt = parse_date(txn["date"], year_hint, statement_end_month=end_month)
            category = categorize_transaction(txn)
            all_transactions.append({
                **txn,
                "parsed_date": dt,
                "month_key": dt.strftime("%Y-%m") if dt else "unknown",
                "category": category,
                "source_file": f.filename,
            })

    all_transactions.sort(key=lambda t: t["parsed_date"] or datetime.min)
    summary = compute_summary(all_transactions)
    # Strip non-serializable fields
    clean_txns = [{k: v for k, v in t.items() if k != "parsed_date"} for t in all_transactions]
    return {"summary": summary, "transactions": clean_txns}


SPENDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spending Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f7; color: #1d1d1f; line-height: 1.5; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
  header { text-align: center; margin-bottom: 32px; }
  header h1 { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: #86868b; font-size: 14px; }

  .upload-area { background: white; border: 2px dashed #ccc; border-radius: 12px; padding: 48px; text-align: center; cursor: pointer; transition: border-color 0.2s; margin-bottom: 24px; }
  .upload-area:hover, .upload-area.dragover { border-color: #007aff; background: #f0f7ff; }
  .upload-area input { display: none; }
  .upload-area p { font-size: 16px; color: #666; }
  .upload-area .icon { font-size: 48px; margin-bottom: 12px; }
  .file-list { margin-top: 12px; font-size: 13px; color: #555; }
  .file-list span { background: #e8f0fe; padding: 3px 10px; border-radius: 12px; margin: 2px; display: inline-block; }
  .btn { background: #007aff; color: white; border: none; padding: 14px 32px; border-radius: 8px; font-size: 16px; cursor: pointer; margin-top: 16px; }
  .btn:disabled { background: #ccc; cursor: default; }
  .btn:hover:not(:disabled) { background: #0056b3; }

  .spinner { display: none; margin: 24px auto; text-align: center; }
  .spinner.active { display: block; }
  .spinner .dot { display: inline-block; width: 10px; height: 10px; background: #007aff; border-radius: 50%; margin: 0 4px; animation: bounce 1.4s infinite ease-in-out both; }
  .spinner .dot:nth-child(1) { animation-delay: -0.32s; }
  .spinner .dot:nth-child(2) { animation-delay: -0.16s; }
  @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
  .progress-text { text-align: center; color: #86868b; font-size: 14px; margin-top: 8px; }

  #report { display: none; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 32px; }
  .card { background: #fff; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .card .label { font-size: 12px; color: #86868b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .card .value { font-size: 22px; font-weight: 700; }
  .positive { color: #34c759; }
  .negative { color: #ff3b30; }

  .chart-section { background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .chart-section h2 { font-size: 18px; font-weight: 600; margin-bottom: 16px; }
  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
  @media (max-width: 768px) { .chart-row { grid-template-columns: 1fr; } }

  .table-section { background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow-x: auto; }
  .table-section h2 { font-size: 18px; font-weight: 600; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th { text-align: left; padding: 10px 12px; border-bottom: 2px solid #e5e5ea; color: #86868b; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  td { padding: 10px 12px; border-bottom: 1px solid #f2f2f7; }
  tr:hover { background: #f9f9fb; }
  .money { font-variant-numeric: tabular-nums; text-align: right; }
  th.money { text-align: right; }
  footer { text-align: center; color: #86868b; font-size: 12px; margin-top: 32px; padding-bottom: 24px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Spending Analysis</h1>
    <p class="subtitle">Upload bank statement PDFs to analyze spending patterns</p>
  </header>

  <div class="upload-area" id="dropzone">
    <div class="icon">&#128200;</div>
    <p id="dropLabel">Drop bank statement PDFs here or click to upload</p>
    <p style="font-size:13px;color:#999;margin-top:8px;">Upload multiple months for best results</p>
    <input type="file" id="fileInput" accept=".pdf" multiple>
    <div class="file-list" id="fileList"></div>
  </div>
  <div style="text-align:center;">
    <button class="btn" id="analyzeBtn" disabled onclick="analyze()">Analyze Spending</button>
  </div>

  <div class="spinner" id="spinner"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
  <p class="progress-text" id="progressText"></p>

  <div id="report">
    <div class="summary-grid" id="summaryCards"></div>
    <div class="chart-section">
      <h2>Monthly Income vs Expenses</h2>
      <canvas id="monthlyChart" height="100"></canvas>
    </div>
    <div class="chart-row">
      <div class="chart-section">
        <h2>Spending by Category</h2>
        <canvas id="categoryChart"></canvas>
      </div>
      <div class="chart-section" id="balanceSection" style="display:none;">
        <h2>Account Balance Trend</h2>
        <canvas id="balanceChart"></canvas>
      </div>
    </div>
    <div class="table-section">
      <h2>Monthly Breakdown</h2>
      <table><thead><tr><th>Month</th><th class="money">Income</th><th class="money">Expenses</th><th class="money">Net</th><th>Txns</th></tr></thead><tbody id="monthlyBody"></tbody></table>
    </div>
    <div class="table-section">
      <h2>Top Merchants</h2>
      <table><thead><tr><th>Merchant</th><th>Transactions</th><th class="money">Total Spent</th></tr></thead><tbody id="merchantBody"></tbody></table>
    </div>
    <footer>Powered by LayoutLMv3 Statement Parser</footer>
  </div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const fileList = document.getElementById('fileList');
let selectedFiles = [];

dropzone.addEventListener('click', () => fileInput.click());
['dragover','dragenter'].forEach(e => dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.add('dragover'); }));
['dragleave','drop'].forEach(e => dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.remove('dragover'); }));
dropzone.addEventListener('drop', ev => {
  const files = Array.from(ev.dataTransfer.files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
  addFiles(files);
});
fileInput.addEventListener('change', () => { addFiles(Array.from(fileInput.files)); });

function addFiles(files) {
  selectedFiles = [...selectedFiles, ...files];
  fileList.innerHTML = selectedFiles.map(f => '<span>' + f.name + '</span>').join(' ');
  document.getElementById('dropLabel').textContent = selectedFiles.length + ' PDF(s) selected';
  analyzeBtn.disabled = selectedFiles.length === 0;
}

function fmt(val) {
  const sign = val < 0 ? '-' : '';
  return sign + '$' + Math.abs(val).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
}

async function analyze() {
  analyzeBtn.disabled = true;
  document.getElementById('spinner').classList.add('active');
  document.getElementById('progressText').textContent = 'Parsing ' + selectedFiles.length + ' statement(s)... This may take a minute.';
  document.getElementById('report').style.display = 'none';

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append('files', f));

  try {
    const resp = await fetch('/v1/analyze', { method: 'POST', body: formData });
    if (!resp.ok) throw new Error('Server error: ' + resp.status);
    const data = await resp.json();
    renderReport(data.summary, data.transactions);
  } catch(e) {
    alert('Error: ' + e.message);
  }

  document.getElementById('spinner').classList.remove('active');
  document.getElementById('progressText').textContent = '';
  analyzeBtn.disabled = false;
}

function renderReport(summary, transactions) {
  document.getElementById('report').style.display = 'block';

  // Summary cards
  const netClass = summary.net_cashflow >= 0 ? 'positive' : 'negative';
  document.getElementById('summaryCards').innerHTML = `
    <div class="card"><div class="label">Total Income</div><div class="value positive">${fmt(summary.total_income)}</div></div>
    <div class="card"><div class="label">Total Expenses</div><div class="value negative">${fmt(summary.total_expenses)}</div></div>
    <div class="card"><div class="label">Net Cashflow</div><div class="value ${netClass}">${fmt(summary.net_cashflow)}</div></div>
    <div class="card"><div class="label">Avg Monthly Spend</div><div class="value">${fmt(summary.avg_monthly_spend)}</div></div>
    <div class="card"><div class="label">Transactions</div><div class="value">${summary.total_transactions.toLocaleString()}</div></div>
    <div class="card"><div class="label">Months</div><div class="value">${summary.months_analyzed}</div></div>
  `;

  // Month labels
  const monthLabels = summary.months.map(m => {
    const [y, mo] = m.split('-');
    const dt = new Date(parseInt(y), parseInt(mo)-1);
    return dt.toLocaleString('default', {month: 'short', year: 'numeric'});
  });
  const incomeData = summary.months.map(m => summary.monthly_income[m] || 0);
  const expenseData = summary.months.map(m => summary.monthly_expenses[m] || 0);

  // Monthly bar chart
  new Chart(document.getElementById('monthlyChart'), {
    type: 'bar',
    data: {
      labels: monthLabels,
      datasets: [
        { label: 'Income', data: incomeData, backgroundColor: '#34c759', borderRadius: 4 },
        { label: 'Expenses', data: expenseData, backgroundColor: '#ff3b30', borderRadius: 4 },
      ]
    },
    options: {
      responsive: true,
      plugins: { tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + fmt(ctx.parsed.y) } } },
      scales: { y: { beginAtZero: true, ticks: { callback: v => '$' + v.toLocaleString() } } }
    }
  });

  // Category doughnut
  const catLabels = Object.keys(summary.category_totals);
  const catData = Object.values(summary.category_totals);
  const catColors = ['#ff6384','#36a2eb','#ffce56','#4bc0c0','#9966ff','#ff9f40','#c9cbcf','#7bc043','#f37735','#00aba9'];
  new Chart(document.getElementById('categoryChart'), {
    type: 'doughnut',
    data: { labels: catLabels, datasets: [{ data: catData, backgroundColor: catColors.slice(0, catLabels.length), borderWidth: 2, borderColor: '#fff' }] },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'right', labels: { font: { size: 12 } } },
        tooltip: { callbacks: { label: ctx => {
          const total = ctx.dataset.data.reduce((a,b) => a+b, 0);
          return ctx.label + ': ' + fmt(ctx.parsed) + ' (' + ((ctx.parsed/total)*100).toFixed(1) + '%)';
        }}}
      }
    }
  });

  // Balance trend
  const balances = summary.daily_balances || [];
  if (balances.length >= 5) {
    document.getElementById('balanceSection').style.display = 'block';
    new Chart(document.getElementById('balanceChart'), {
      type: 'line',
      data: { labels: balances.map(b => b[0]), datasets: [{ label: 'Balance', data: balances.map(b => b[1]), borderColor: '#007aff', backgroundColor: 'rgba(0,122,255,0.1)', fill: true, tension: 0.3, pointRadius: 1 }] },
      options: {
        responsive: true,
        plugins: { tooltip: { callbacks: { label: ctx => fmt(ctx.parsed.y) } } },
        scales: { y: { ticks: { callback: v => '$' + v.toLocaleString() } }, x: { ticks: { maxTicksLimit: 12 } } }
      }
    });
  }

  // Monthly table
  document.getElementById('monthlyBody').innerHTML = summary.months.map((m, i) => {
    const inc = summary.monthly_income[m] || 0;
    const exp = summary.monthly_expenses[m] || 0;
    const net = inc - exp;
    const txns = summary.monthly_txn_count[m] || 0;
    return `<tr><td>${monthLabels[i]}</td><td class="money">${fmt(inc)}</td><td class="money">${fmt(exp)}</td><td class="money ${net>=0?'positive':'negative'}">${fmt(net)}</td><td>${txns}</td></tr>`;
  }).join('');

  // Merchants table
  document.getElementById('merchantBody').innerHTML = (summary.top_merchants || []).map(([name, data]) =>
    `<tr><td>${name}</td><td>${data.count}</td><td class="money">${fmt(data.total)}</td></tr>`
  ).join('');

  // Scroll to report
  document.getElementById('report').scrollIntoView({ behavior: 'smooth' });
}
</script>
</body></html>"""


@app.get("/spending", response_class=HTMLResponse)
async def spending_ui():
    return SPENDING_HTML


@app.get("/", response_class=HTMLResponse)
async def ui_home():
    return UPLOAD_HTML


@app.post("/v1/upload")
async def upload_parse(file: UploadFile = File(...)):
    """Parse an uploaded PDF file."""
    pdf_bytes = await file.read()
    b64 = base64.b64encode(pdf_bytes).decode()
    req = ParseRequest(pdf_base64=b64)
    return await parse_statement(req)


@app.get("/v1/feedback/flagged")
async def feedback_flagged():
    """Return flagged low-confidence extractions grouped by bank format."""
    from pathlib import Path as P
    feedback_dir = P(__file__).resolve().parent / "data" / "feedback"
    if not feedback_dir.exists():
        return {"banks": {}, "total_flagged": 0}

    by_bank = {}
    total = 0
    for fp in sorted(feedback_dir.glob("*.json")):
        with open(fp) as f:
            rec = json.load(f)
        bank = rec.get("bank_format", "unknown")
        if bank not in by_bank:
            by_bank[bank] = {"count": 0, "avg_confidence": [], "pages": []}
        by_bank[bank]["count"] += 1
        by_bank[bank]["avg_confidence"].append(rec.get("avg_confidence", 0))
        by_bank[bank]["pages"].append({
            "pdf_hash": rec.get("pdf_hash"),
            "flagged_transactions": rec.get("flagged_transactions", 0),
            "total_transactions": rec.get("total_transactions", 0),
            "avg_confidence": rec.get("avg_confidence", 0),
        })
        total += 1

    # Compute averages and sort by lowest confidence
    for bank in by_bank:
        confs = by_bank[bank]["avg_confidence"]
        by_bank[bank]["avg_confidence"] = round(sum(confs) / len(confs), 4) if confs else 0

    sorted_banks = dict(sorted(by_bank.items(), key=lambda x: x[1]["avg_confidence"]))
    return {"banks": sorted_banks, "total_flagged": total}


# --- Retraining ---

_retrain_status = {"running": False, "last_run": None, "last_result": None}
_retrain_lock = threading.Lock()


@app.post("/v1/retrain")
async def trigger_retrain(epochs: int = 3, promote_if_better: bool = True):
    """Trigger model retraining with auto-labeled + existing training data.

    Pipeline:
      1. Run align_labels.py to convert auto-labeled Claude Vision data → BIO tags
      2. Run retrain.py to train candidate model
      3. Compare candidate vs current model
      4. Promote if candidate F1 >= current F1 (when promote_if_better=True)

    Runs in a background thread — returns immediately.
    """
    with _retrain_lock:
        if _retrain_status["running"]:
            return {
                "status": "already_running",
                "started_at": _retrain_status["last_run"],
            }
        _retrain_status["running"] = True
        _retrain_status["last_run"] = datetime.now(timezone.utc).isoformat()

    def _run_retrain():
        steps_completed = []
        try:
            # Step 1: Align all labeled data (including auto-labeled)
            logger.info("Retrain step 1: aligning labels...")
            r = subprocess.run(
                [sys.executable, "align_labels.py"],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                timeout=300,
            )
            steps_completed.append(f"align: rc={r.returncode}")
            if r.returncode != 0:
                _retrain_status["last_result"] = f"align failed: {r.stderr.decode()[:300]}"
                return

            # Step 2: Retrain candidate model
            logger.info(f"Retrain step 2: training candidate ({epochs} epochs)...")
            r = subprocess.run(
                [sys.executable, "retrain.py", "--skip-export", "--epochs", str(epochs)],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                timeout=1800,
            )
            steps_completed.append(f"train: rc={r.returncode}")
            if r.returncode != 0:
                _retrain_status["last_result"] = f"train failed: {r.stderr.decode()[:300]}"
                return

            # Step 3: Promote if better
            if promote_if_better:
                logger.info("Retrain step 3: promoting if better...")
                r = subprocess.run(
                    [sys.executable, "retrain.py", "--promote"],
                    cwd=str(PROJECT_DIR),
                    capture_output=True,
                    timeout=120,
                )
                steps_completed.append(f"promote: rc={r.returncode}")

                # Reload model if promotion succeeded
                if r.returncode == 0 and b"promoted" in r.stdout.lower():
                    try:
                        from predict import reload_parser
                        reload_parser()
                        steps_completed.append("reload: ok")
                        logger.info("Model reloaded after promotion")
                    except Exception as e:
                        steps_completed.append(f"reload: {e}")

            _retrain_status["last_result"] = f"success: {', '.join(steps_completed)}"
            logger.info(f"Retrain complete: {steps_completed}")

        except subprocess.TimeoutExpired:
            _retrain_status["last_result"] = f"timeout after steps: {steps_completed}"
            logger.error("Retrain timed out")
        except Exception as e:
            _retrain_status["last_result"] = f"error: {e}"
            logger.error(f"Retrain error: {e}")
        finally:
            _retrain_status["running"] = False

    thread = threading.Thread(target=_run_retrain, daemon=True)
    thread.start()

    return {
        "status": "started",
        "started_at": _retrain_status["last_run"],
        "epochs": epochs,
        "promote_if_better": promote_if_better,
    }


@app.get("/v1/retrain/status")
async def retrain_status():
    """Check status of the last/current retrain run."""
    return _retrain_status


@app.get("/v1/feedback/stats")
async def feedback_stats():
    """Show retraining readiness: auto-labeled pages, flagged PDFs, etc."""
    labels_dir = PROJECT_DIR / "data" / "labels"
    training_dir = PROJECT_DIR / "data" / "training"
    feedback_dir = PROJECT_DIR / "data" / "feedback"

    # Count auto-labeled pages (from Claude Vision fallback)
    auto_stems = []
    if labels_dir.exists():
        auto_stems = [d.name for d in labels_dir.iterdir()
                      if d.is_dir() and d.name.startswith("auto_")]
    auto_pages = sum(
        len(list((labels_dir / s).glob("page_*_labels.json")))
        for s in auto_stems
    )

    # Count aligned training pages (BIO files generated from auto-labels)
    auto_aligned = sum(
        len(list((training_dir / s).glob("page_*_bio.json")))
        for s in auto_stems
        if (training_dir / s).exists()
    )

    # Count all training data
    total_training_stems = []
    if training_dir.exists():
        total_training_stems = [d.name for d in training_dir.iterdir() if d.is_dir()]
    total_training_pages = sum(
        len(list((training_dir / s).glob("page_*_bio.json")))
        for s in total_training_stems
    )

    # Count flagged feedback files
    flagged_count = len(list(feedback_dir.glob("*.json"))) if feedback_dir.exists() else 0

    ready_for_retrain = auto_pages >= MIN_PAGES_FOR_RETRAIN

    return {
        "auto_labeled_pdfs": len(auto_stems),
        "auto_labeled_pages": auto_pages,
        "auto_aligned_pages": auto_aligned,
        "total_training_pages": total_training_pages,
        "flagged_pdfs": flagged_count,
        "min_pages_for_retrain": MIN_PAGES_FOR_RETRAIN,
        "ready_for_retrain": ready_for_retrain,
        "retrain_status": _retrain_status,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
