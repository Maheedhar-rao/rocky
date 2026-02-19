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

EXTRACTION_PROMPT = """You are analyzing a bank statement page. Extract ALL transactions visible on this page.

For each transaction, return:
- date: the transaction date as shown (e.g., "01/15", "Jan 15", "2024-01-15")
- description: the full transaction description text
- amount: the dollar amount as a number (positive for deposits/credits, negative for withdrawals/debits)
- type: "credit" or "debit"
- balance_after: the running balance after this transaction (if shown), or null

Also extract page metadata:
- bank_name: the bank name if visible
- account_number_last4: last 4 digits of account if visible
- statement_period: the statement date range if visible
- page_number: the page number if visible

Return ONLY valid JSON in this exact format:
{
  "metadata": {
    "bank_name": "...",
    "account_number_last4": "...",
    "statement_period": "...",
    "page_number": null
  },
  "transactions": [
    {
      "date": "01/15",
      "description": "DIRECT DEPOSIT EMPLOYER NAME",
      "amount": 2500.00,
      "type": "credit",
      "balance_after": 5000.00
    }
  ],
  "has_transactions": true,
  "notes": "any issues or ambiguities"
}

If this page has no transactions (e.g., summary page, cover page), set has_transactions=false and transactions=[].
Be thorough — extract EVERY transaction row, including small fees and interest."""

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
        max_tokens=4096,
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
                    "text": EXTRACTION_PROMPT,
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
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdf_stems = [args.pdf]
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

    print(f"\nDone. Labeled {total_labeled} pages, {total_txns} transactions found, {total_errors} errors")


if __name__ == "__main__":
    main()
