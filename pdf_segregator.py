#!/usr/bin/env python3
"""
Multi-PDF segregator for the spending-analysis flow.

The original /v1/analyze flow parsed PDFs one at a time in a sequential loop and
treated each transaction's page index as PDF-local. When several statements are
uploaded together, those page indices collide (every PDF restarts at page 0), so
parallel results can't be reassembled or traced.

This module adds the missing PDF dimension and a parallel dispatch layer:

  1. PLAN     — count pages per PDF (pdfplumber metadata, ~ms) to size the pool
  2. DISPATCH — fan out parse_pdf across PDFs on a bounded thread pool
  3. STAMP    — tag every transaction with a provenance triple (pdf_idx, page, row)
                plus its own PDF's year context, so the global page id is unique
                ("O1, O2 ..." = (pdf 0, page 1), (pdf 0, page 2) ...)
  4. REGROUP  — sort by the provenance triple (deterministic), then fold by
                month_key (date-derived) via the existing compute_summary

Provenance (pdf_idx, page, row) reassembles the *work*; month_key groups the
*analytics*. They are intentionally orthogonal — a single page can hold two
months, and one month can span multiple PDFs.
"""

import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pdfplumber

from bank_patterns import detect_bank
from summary_reconcile import reconcile_summary

MAX_PARSE_WORKERS = int(os.environ.get("MAX_PARSE_WORKERS", "4"))


@dataclass
class PdfPlan:
    """Cheap pre-pass result for one uploaded PDF."""
    pdf_idx: int
    filename: str
    pages: int
    nbytes: int
    bank_name: str = "unknown"


@dataclass
class SegregationResult:
    transactions: list = field(default_factory=list)  # flat, provenance-stamped
    summary: dict = field(default_factory=dict)
    plan: list = field(default_factory=list)          # list[PdfPlan]
    timings_ms: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)         # pdf_idx -> error string
    reconciliation: dict = field(default_factory=dict)  # running-balance accuracy (global)
    verdicts: dict = field(default_factory=dict)        # pdf_idx -> per-statement trust verdict


def count_pages(pdf_bytes: bytes) -> int:
    """Page count from PDF metadata only — no rasterization. ~ms even for big PDFs."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return len(pdf.pages)


def _count_and_bank(pdf_bytes: bytes) -> tuple:
    """One open: page count (metadata) + bank name (page-1 text). Both ~ms, no ML."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n = len(pdf.pages)
        first_text = (pdf.pages[0].extract_text() or "") if pdf.pages else ""
    return n, detect_bank(first_text)


def plan_uploads(pdfs: list) -> list:
    """PLAN phase. pdfs = list of (filename, bytes). Returns list[PdfPlan]."""
    plan = []
    for idx, (name, blob) in enumerate(pdfs):
        try:
            n, bank = _count_and_bank(blob)
        except Exception:
            n, bank = 0, "unknown"
        plan.append(PdfPlan(pdf_idx=idx, filename=name, pages=n, nbytes=len(blob), bank_name=bank))
    return plan


BANK_RESOLVE_MODEL = os.environ.get("BANK_RESOLVE_MODEL", "claude-haiku-4-5-20251001")


def resolve_bank_with_claude(pdf_bytes: bytes) -> str:
    """Guaranteed-name fallback: ask Claude to name the institution from page 1.

    Only meant for the tail the free tiers can't classify. Renders page 1 only and
    uses a cheap model + tiny prompt, so cost is negligible. Returns 'unknown' if
    Claude can't tell or anything errors (never raises).
    """
    try:
        import base64
        import anthropic
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)
        if not images:
            return "unknown"
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=BANK_RESOLVE_MODEL,
            max_tokens=32,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": "What bank or financial institution issued this "
                     "statement? Reply with ONLY the institution name (e.g. 'Chase', "
                     "'FourLeaf Federal Credit Union'). If you cannot tell, reply 'unknown'."},
                ],
            }],
        )
        name = (resp.content[0].text or "").strip().strip(".\"'")
        return name if name and name.lower() != "unknown" and len(name) <= 60 else "unknown"
    except Exception:
        return "unknown"


def resolve_unknown_banks(plan: list, pdfs: list) -> int:
    """For every plan entry still 'unknown', escalate to Claude. Mutates plan.

    Returns how many were resolved. Cheap: only runs on the free-tier misses.
    """
    resolved = 0
    for p in plan:
        if p.bank_name and p.bank_name != "unknown":
            continue
        name = resolve_bank_with_claude(pdfs[p.pdf_idx][1])
        if name != "unknown":
            p.bank_name = name
            resolved += 1
    return resolved


def _infer_year_context(filename: str, pdf_bytes: bytes):
    """Per-PDF year inference — must stay per-PDF so month_key is correct.

    Mirrors the logic in /v1/analyze: filename hint first, then PDF content.
    """
    from spending_demo import infer_year_from_pdf

    year_hint, end_month = None, None
    m = re.search(r"20[2-3]\d", filename or "")
    if m:
        year_hint = int(m.group())
    if not year_hint:
        year_hint, end_month = infer_year_from_pdf(pdf_bytes)
    return year_hint, end_month


def _stamp_provenance(raw_txns: list, plan: PdfPlan, year_hint, end_month) -> list:
    """STAMP phase for one PDF's transactions.

    Adds the provenance triple (pdf_idx, page, row) + global page id, and the
    date-derived month_key/category, using THIS PDF's own year context.
    """
    from spending_demo import categorize_transaction, parse_date

    # row index is per (pdf, page); parse_pdf returns rows in reading order.
    row_by_page = {}
    enriched = []
    for txn in raw_txns:
        page = txn.get("page", 0)
        row = row_by_page.get(page, 0)
        row_by_page[page] = row + 1

        dt = parse_date(txn.get("date", ""), year_hint, statement_end_month=end_month)
        enriched.append({
            **txn,
            "pdf_idx": plan.pdf_idx,
            "source_file": plan.filename,
            "bank_name": plan.bank_name,
            "page": page,
            "row": row,
            "page_uid": f"{plan.pdf_idx}:{page}",   # the "O1, O2" imaginary number
            "parsed_date": dt,
            "month_key": dt.strftime("%Y-%m") if dt else "unknown",
            "category": categorize_transaction(txn),
        })
    return enriched


def _parse_one(plan: PdfPlan, pdf_bytes: bytes) -> list:
    """DISPATCH worker: parse one PDF and stamp its transactions (self-contained)."""
    from predict import get_parser

    parser = get_parser()                       # shared singleton, one VRAM copy
    result = parser.parse_pdf(pdf_bytes)
    year_hint, end_month = _infer_year_context(plan.filename, pdf_bytes)
    return _stamp_provenance(result.transactions, plan, year_hint, end_month)


def reconcile_running_balance(transactions: list) -> dict:
    """Verify extracted amounts reproduce the bank's printed running balance.

    The gold-standard intrinsic check: for each PDF, walk transactions in
    (page, row) order. Each step's balance delta must equal the signed amount
    (credit = +amount, debit = -amount). Steps that don't match are flagged.

    Catches: sign errors (e.g. Credit Interest tagged debit), missing/extra
    rows (balance jumps), wrong amounts, and phantom non-transaction rows
    (Starting Balance) that break continuity. Needs no ground-truth labels —
    the printed Balance column IS the ground truth.

    Note: legitimate reversal pairs ("(Rejected)") post to the running balance,
    so they PASS here; their effect on gross deposit/withdrawal totals is a
    separate summary-box reconciliation.
    """
    by_pdf = {}
    for t in transactions:
        by_pdf.setdefault(t["pdf_idx"], []).append(t)

    per_pdf = {}
    all_flags = []
    for pdf_idx, txns in by_pdf.items():
        txns = sorted(txns, key=lambda t: (t["page"], t["row"]))
        prev_bal = None
        checks = ok = no_balance = 0
        flags = []
        for t in txns:
            bal = t.get("balance_after")
            if bal is None:
                no_balance += 1
                continue
            signed = t["amount"] if t.get("type") == "credit" else -t["amount"]
            if prev_bal is not None:
                checks += 1
                expected = round(prev_bal + signed, 2)
                if abs(expected - bal) < 0.01:
                    ok += 1
                else:
                    flags.append({
                        "pdf_idx": pdf_idx,
                        "page_uid": t.get("page_uid"),
                        "row": t.get("row"),
                        "date": t.get("date"),
                        "description": (t.get("description") or "")[:40],
                        "signed_amount": round(signed, 2),
                        "prev_balance": round(prev_bal, 2),
                        "balance_after": round(bal, 2),
                        "expected": expected,
                        "delta": round(bal - expected, 2),
                    })
            prev_bal = bal
        per_pdf[pdf_idx] = {
            "checks": checks,
            "ok": ok,
            "rate": round(ok / checks, 4) if checks else None,
            "no_balance": no_balance,
            "flagged": len(flags),
        }
        all_flags.extend(flags)

    total_checks = sum(p["checks"] for p in per_pdf.values())
    total_ok = sum(p["ok"] for p in per_pdf.values())
    return {
        "per_pdf": per_pdf,
        "overall_rate": round(total_ok / total_checks, 4) if total_checks else None,
        "total_checks": total_checks,
        "total_ok": total_ok,
        "flagged": all_flags,
    }


def extract_summary_totals_with_claude(pdf_bytes: bytes) -> dict:
    """Page-1 control totals for the summary-box anchor: beginning/ending balance,
    total deposits, total withdrawals. Any-bank via Claude. {} on failure.
    """
    try:
        import base64
        import json as _json
        import anthropic
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)
        if not images:
            return {}
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=BANK_RESOLVE_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text":
                        "From this bank statement page, extract the statement control totals as JSON:\n"
                        '{"beginning_balance": number, "ending_balance": number, '
                        '"total_deposits": number, "total_withdrawals": number}\n'
                        "total_deposits = total of all credits/deposits. total_withdrawals = total of ALL "
                        "debits/withdrawals/checks/fees combined, as a POSITIVE number. Use null for any "
                        "value not printed. Reply with ONLY the JSON object."},
                ],
            }],
        )
        text = (resp.content[0].text or "").strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.splitlines() if not l.startswith("```"))
        d = _json.loads(text)
        return {k: d.get(k) for k in
                ("beginning_balance", "ending_balance", "total_deposits", "total_withdrawals")}
    except Exception:
        return {}


def verify_statement(txns: list, totals: dict) -> dict:
    """Combine both anchors into one trust verdict for a single statement.

    running-balance (per-row) + summary-box (control totals) →
      reconciled    every anchor that has data passes
      needs_review  some anchor that has data FAILS (a real discrepancy)
      unverified    no anchor has enough data to judge (route to escalation)

    A statement that passes one anchor but fails another is needs_review — a
    mismatch anywhere means we don't silently ship it.
    """
    # --- running-balance anchor (single statement) ---
    rb = reconcile_running_balance(txns)
    rb_checks = rb["total_checks"]
    rb_nobal = sum(p["no_balance"] for p in rb["per_pdf"].values())
    rb_coverage = rb_checks / max(1, rb_checks + rb_nobal)
    rb_rate = rb["overall_rate"]
    if rb_checks >= 5 and rb_coverage >= 0.5:
        rb_status = "pass" if (rb_rate is not None and rb_rate >= 0.99) else "fail"
    else:
        rb_status = "n/a"  # too few printed balances to judge

    # --- summary-box anchor ---
    sb = reconcile_summary(
        txns,
        beginning=totals.get("beginning_balance"),
        ending=totals.get("ending_balance"),
        stated_total_deposits=totals.get("total_deposits"),
        stated_total_withdrawals=totals.get("total_withdrawals"),
    )
    sb_status = ("pass" if sb.reconciled else "fail") if sb.checks else "n/a"

    statuses = [s for s in (rb_status, sb_status) if s != "n/a"]
    if not statuses:
        verdict = "unverified"
    elif "fail" in statuses:
        verdict = "needs_review"
    else:
        verdict = "reconciled"

    return {
        "verdict": verdict,
        "anchors": {
            "running_balance": {
                "status": rb_status, "rate": rb_rate,
                "coverage": round(rb_coverage, 3), "checks": rb_checks,
            },
            "summary_box": {
                "status": sb_status, "reason": sb.reason, "checks": sb.checks,
            },
        },
        "flagged_rows": rb["flagged"][:50],
    }


def segregate_and_analyze(pdfs: list, max_workers: int = None,
                          resolve_banks: bool = True,
                          verify: bool = True) -> SegregationResult:
    """End-to-end: plan → parallel parse → stamp → sort → regroup by month.

    pdfs = list of (filename, bytes). Returns a SegregationResult.
    resolve_banks: escalate free-tier 'unknown' banks to Claude (guaranteed name).
    """
    from spending_demo import compute_summary

    t0 = time.time()
    plan = plan_uploads(pdfs)
    if resolve_banks:
        resolve_unknown_banks(plan, pdfs)   # Claude fallback for the 'unknown' tail
    t_plan = (time.time() - t0) * 1000

    workers = max_workers or min(len(pdfs), MAX_PARSE_WORKERS)
    workers = max(1, workers)

    # DISPATCH — bounded thread pool, sharing the model singleton.
    # One PDF failing (e.g. an unreadable scan) must not sink the whole batch.
    t1 = time.time()
    errors = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_parse_one, p, pdfs[p.pdf_idx][1]): p for p in plan}
        results_by_idx = {}
        for fut, p in futures.items():
            try:
                results_by_idx[p.pdf_idx] = fut.result()
            except Exception as e:  # noqa: BLE001 — surface, don't crash the batch
                results_by_idx[p.pdf_idx] = []
                errors[p.pdf_idx] = f"{type(e).__name__}: {e}"
        per_pdf = [results_by_idx[p.pdf_idx] for p in plan]
    t_parse = (time.time() - t1) * 1000

    # REGROUP — merge, then sort by the provenance triple for determinism
    # (fixes daily_balances last-write-wins under out-of-order parallel results),
    # then compute_summary folds by month_key.
    t2 = time.time()
    merged = [txn for chunk in per_pdf for txn in chunk]
    merged.sort(key=lambda t: (t["pdf_idx"], t["page"], t["row"]))
    summary = compute_summary(merged)
    reconciliation = reconcile_running_balance(merged)
    t_regroup = (time.time() - t2) * 1000

    # VERIFY — per-statement trust verdict (both anchors). Summary-box totals come
    # from Claude (page-1, cheap). Skipped for PDFs that errored out in parsing.
    t3 = time.time()
    verdicts = {}
    if verify:
        for p in plan:
            if p.pdf_idx in errors:
                continue
            txns = results_by_idx.get(p.pdf_idx, [])
            totals = extract_summary_totals_with_claude(pdfs[p.pdf_idx][1])
            verdicts[p.pdf_idx] = verify_statement(txns, totals)
    t_verify = (time.time() - t3) * 1000

    # Strip the non-serializable parsed_date before returning over the wire.
    clean = [{k: v for k, v in t.items() if k != "parsed_date"} for t in merged]

    return SegregationResult(
        transactions=clean,
        summary=summary,
        plan=plan,
        timings_ms={
            "plan": round(t_plan, 1),
            "parse": round(t_parse, 1),
            "regroup": round(t_regroup, 1),
            "verify": round(t_verify, 1),
            "total": round((time.time() - t0) * 1000, 1),
            "workers": workers,
        },
        errors=errors,
        reconciliation=reconciliation,
        verdicts=verdicts,
    )
