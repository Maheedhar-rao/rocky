#!/usr/bin/env python3
"""
Label bank statement pages using Claude Vision.

Sends each page image to Claude Sonnet and gets structured transaction JSON.
Cost: ~$0.003/page × ~1500 pages ≈ $5 total.

Usage:
    python label_with_claude.py                   # label all unlabeled pages
    python label_with_claude.py --limit 10        # label 10 PDFs
    python label_with_claude.py --pdf app_123     # label specific PDF stem
    python label_with_claude.py --dry-run         # count pages without calling API
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

PAGES_DIR = PROJECT_DIR / "data" / "pages"
LABELS_DIR = PROJECT_DIR / "data" / "labels"

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

LABEL_PROMPT = """You are generating training labels for a bank statement parser.

Your job is to identify and extract every transaction from this bank statement page image so it can be used as ground truth training data for a LayoutLMv3 model.

EXTRACT EVERY TRANSACTION — include all of these:
- ACH credits and debits
- POS purchases and debit card transactions
- ATM withdrawals
- Wire transfers (in and out)
- Internal transfers between accounts
- NSF fees, returned item fees, overdraft fees
- Service charges, monthly fees
- Tax payments, loan payments
- Dividend credits

AMOUNT RULES — these must be exact:
- Debits, withdrawals, fees, payments → NEGATIVE (e.g. -253.34)
- Credits, deposits → POSITIVE (e.g. 1250.00)
- Read from the Withdrawal or Deposit column only
- Never read from the Balance column for the amount
- MCA lender payments (Arya Capital, CCS, Trueadvance, Mission Fin, Palmera) are always negative and typically $100–$2000
- NSF fees are typically -29.00

BALANCE:
- running_balance is the exact number in the Balance column on that row
- null if no balance printed on that row

WHAT NOT TO EXTRACT:
- Daily balance summary rows
- Account summary / opening / closing balance rows
- Section headers or totals rows
- Any row that is not an individual transaction

DATE FORMAT: YYYY-MM-DD always. Infer year from statement header context.

Return only a JSON object in this exact format, no explanation:
{
  "has_transactions": true,
  "statement_year": 2025,
  "statement_month": 12,
  "transactions": [
    {
      "date": "2025-12-03",
      "description": "ACH Paid From Optix LLC Payroll",
      "amount": 1250.00,
      "type": "credit",
      "running_balance": 3842.17
    },
    {
      "date": "2025-12-04",
      "description": "POS Debit Arya Capital 8887733199",
      "amount": -253.34,
      "type": "debit",
      "running_balance": 3588.83
    }
  ],
  "notes": "any observations about this page layout or edge cases"
}

If this page has no transactions (e.g. it is a cover page or summary page), return:
{
  "has_transactions": false,
  "transactions": [],
  "notes": "reason this page has no transactions"
}"""

# Singleton Anthropic client
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def label_page(image_path: Path) -> dict:
    """Send a page image to Claude Vision and get transaction labels."""
    client = _get_client()

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": LABEL_PROMPT,
                },
            ],
        }],
    )

    text = response.content[0].text.strip()

    # Extract JSON from response (handle markdown code blocks)
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {
            "metadata": {},
            "transactions": [],
            "has_transactions": False,
            "notes": f"JSON parse error. Raw: {text[:500]}",
            "parse_error": True,
        }

    # Add token usage for cost tracking
    result["_usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    return result


def process_pdf_pages(pdf_stem: str, force: bool = False) -> dict:
    """Label all pages for a given PDF."""
    page_dir = PAGES_DIR / pdf_stem
    label_dir = LABELS_DIR / pdf_stem
    label_dir.mkdir(parents=True, exist_ok=True)

    stats = {"pages": 0, "labeled": 0, "transactions": 0, "errors": 0, "skipped": 0}

    page_images = sorted(page_dir.glob("page_*.png"))
    stats["pages"] = len(page_images)

    for img_path in page_images:
        page_idx = img_path.stem.replace("page_", "")
        label_path = label_dir / f"page_{page_idx}_labels.json"

        if label_path.exists() and not force:
            stats["skipped"] += 1
            continue

        try:
            result = label_page(img_path)
            result["_source_image"] = img_path.name
            result["_pdf_stem"] = pdf_stem

            with open(label_path, "w") as f:
                json.dump(result, f, indent=2)

            txn_count = len(result.get("transactions", []))
            stats["transactions"] += txn_count
            stats["labeled"] += 1

            if result.get("parse_error"):
                stats["errors"] += 1

            # Rate limiting: ~50 requests/minute for Sonnet
            time.sleep(0.5)

        except Exception as e:
            print(f"    Error labeling {img_path.name}: {e}")
            stats["errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Label bank statement pages with Claude Vision")
    parser.add_argument("--limit", type=int, help="Max PDFs to label")
    parser.add_argument("--pdf", help="Label specific PDF stem (folder name in data/pages/)")
    parser.add_argument("--force", action="store_true", help="Re-label already labeled pages")
    parser.add_argument("--dry-run", action="store_true", help="Count pages without calling API")
    parser.add_argument("--queue", action="store_true",
                        help="Process only PDFs queued by predict.py (unseen banks, low confidence)")
    parser.add_argument("--relabel-errors", action="store_true",
                        help="Re-label only pages that had parse_error in previous labeling")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    QUEUE_DIR = PROJECT_DIR / "data" / "queue"

    if args.relabel_errors:
        # Find all pages with parse_error and re-label them
        import glob as _glob
        error_pages = []
        for lf in sorted(_glob.glob(str(LABELS_DIR / "**" / "page_*_labels.json"), recursive=True)):
            with open(lf) as f:
                d = json.load(f)
            if d.get("parse_error") or d.get("error"):
                error_pages.append(Path(lf))
        if not error_pages:
            print("No parse-error pages found.")
            return
        print(f"Found {len(error_pages)} parse-error pages to re-label")
        est_cost = len(error_pages) * 0.003
        print(f"Estimated cost: ${est_cost:.2f}")
        if args.dry_run:
            for ep in error_pages:
                print(f"  {ep.relative_to(LABELS_DIR)}")
            return
        total_labeled = 0
        total_txns = 0
        total_errors = 0
        for ep in error_pages:
            stem = ep.parent.name
            page_idx = ep.stem.replace("_labels", "").replace("page_", "")
            img_path = PAGES_DIR / stem / f"page_{page_idx}.png"
            if not img_path.exists():
                print(f"  SKIP {stem}/page_{page_idx} — image not found")
                continue
            print(f"  Re-labeling {stem}/page_{page_idx}...", end=" ")
            try:
                result = label_page(img_path)
                result["_source_image"] = img_path.name
                result["_pdf_stem"] = stem
                with open(ep, "w") as f:
                    json.dump(result, f, indent=2)
                txns = len(result.get("transactions", []))
                total_txns += txns
                total_labeled += 1
                if result.get("parse_error"):
                    total_errors += 1
                    print(f"STILL ERROR")
                else:
                    print(f"{txns} txns")
                time.sleep(0.5)
            except Exception as e:
                print(f"ERROR: {e}")
                total_errors += 1
        print(f"\nDone. Re-labeled {total_labeled} pages, {total_txns} transactions found, {total_errors} errors")
        return

    if args.pdf:
        pdf_stems = [args.pdf]
    elif args.queue:
        # Process only stems queued by predict.py (unseen banks / low confidence)
        if not QUEUE_DIR.exists():
            print("No queued PDFs. Queue is populated when predict.py encounters unseen banks.")
            return
        queued_stems = []
        for qf in sorted(QUEUE_DIR.glob("*.json")):
            with open(qf) as f:
                meta = json.load(f)
            if meta.get("status") == "pending":
                queued_stems.append(meta["stem"])
        if not queued_stems:
            print("No pending queued PDFs.")
            return
        pdf_stems = queued_stems
        print(f"Processing {len(pdf_stems)} queued PDFs (unseen banks / low confidence)")
    else:
        if not PAGES_DIR.exists():
            print(f"No pages directory at {PAGES_DIR}. Run pdf_to_pages.py first.")
            sys.exit(1)
        pdf_stems = sorted([d.name for d in PAGES_DIR.iterdir() if d.is_dir()])

    if args.limit:
        pdf_stems = pdf_stems[:args.limit]

    # Count total work
    total_pages = 0
    unlabeled_pages = 0
    for stem in pdf_stems:
        page_dir = PAGES_DIR / stem
        label_dir = LABELS_DIR / stem
        pages = list(page_dir.glob("page_*.png"))
        total_pages += len(pages)
        for p in pages:
            page_idx = p.stem.replace("page_", "")
            label_path = label_dir / f"page_{page_idx}_labels.json"
            if not label_path.exists() or args.force:
                unlabeled_pages += 1

    est_cost = unlabeled_pages * 0.003
    print(f"PDFs: {len(pdf_stems)}, Total pages: {total_pages}, To label: {unlabeled_pages}")
    print(f"Estimated cost: ${est_cost:.2f}")

    if args.dry_run:
        return

    if unlabeled_pages > 5000:
        print(f"\nThis will make {unlabeled_pages} API calls (~${est_cost:.2f}). Continue? [y/N] ", end="")
        if input().strip().lower() != "y":
            print("Aborted.")
            return

    total_txns = 0
    total_errors = 0
    total_labeled = 0

    for i, stem in enumerate(pdf_stems):
        print(f"\n[{i+1}/{len(pdf_stems)}] {stem}")
        stats = process_pdf_pages(stem, force=args.force)
        total_labeled += stats["labeled"]
        total_txns += stats["transactions"]
        total_errors += stats["errors"]
        print(f"  Pages: {stats['pages']}, Labeled: {stats['labeled']}, "
              f"Txns: {stats['transactions']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")

    # Update queue status for any queued PDFs that were just labeled
    if args.queue and QUEUE_DIR.exists():
        for qf in QUEUE_DIR.glob("*.json"):
            try:
                with open(qf) as f:
                    meta = json.load(f)
                if meta.get("stem") in pdf_stems and meta.get("status") == "pending":
                    meta["status"] = "labeled"
                    meta["labeled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    with open(qf, "w") as f:
                        json.dump(meta, f, indent=2)
            except Exception:
                pass

    print(f"\nDone. Labeled {total_labeled} pages, {total_txns} transactions found, {total_errors} errors")


if __name__ == "__main__":
    main()
