#!/usr/bin/env python3
"""Parallel version of label_with_claude.py — runs N concurrent API calls."""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

PAGES_DIR = PROJECT_DIR / "data" / "pages"
LABELS_DIR = PROJECT_DIR / "data" / "labels"

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
CONCURRENCY = 10  # parallel API calls

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


async def label_page(client, image_path: Path, semaphore):
    """Send a page image to Claude Vision and get transaction labels."""
    async with semaphore:
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        response = await asyncio.to_thread(
            client.messages.create,
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

        # Extract JSON from markdown code blocks
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

        result["_usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return result


async def process_page(client, semaphore, img_path, label_path, pdf_stem, counter):
    """Process a single page."""
    try:
        result = await label_page(client, img_path, semaphore)
        result["_source_image"] = img_path.name
        result["_pdf_stem"] = pdf_stem

        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w") as f:
            json.dump(result, f, indent=2)

        txn_count = len(result.get("transactions", []))
        counter["labeled"] += 1
        counter["transactions"] += txn_count
        if result.get("parse_error"):
            counter["errors"] += 1

        total = counter["labeled"] + counter["errors"]
        if total % 25 == 0:
            print(f"  Progress: {total}/{counter['total']} pages | {counter['transactions']} txns | {counter['errors']} errors")

        return txn_count

    except Exception as e:
        counter["errors"] += 1
        print(f"  Error {img_path.parent.name}/{img_path.name}: {e}")
        return 0


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parallel Claude Vision labeling")
    parser.add_argument("--manifest", help="JSON manifest of PDFs to label (from sample_diverse.py)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    # Determine which PDF stems to label
    if args.manifest:
        manifest = json.load(open(args.manifest))
        target_stems = {p["stem"] for p in manifest["pdfs"]}
        print(f"Manifest: {len(target_stems)} PDFs from {args.manifest}")
    else:
        target_stems = None  # label everything

    # Collect all unlabeled pages
    tasks = []
    for pdf_dir in sorted(PAGES_DIR.iterdir()):
        if not pdf_dir.is_dir():
            continue
        pdf_stem = pdf_dir.name

        if target_stems is not None and pdf_stem not in target_stems:
            continue

        label_dir = LABELS_DIR / pdf_stem

        for img_path in sorted(pdf_dir.glob("page_*.png")):
            page_idx = img_path.stem.replace("page_", "")
            label_path = label_dir / f"page_{page_idx}_labels.json"

            if label_path.exists():
                continue

            tasks.append((img_path, label_path, pdf_stem))

    concurrency = args.concurrency
    print(f"Pages to label: {len(tasks)} (concurrency: {concurrency})")
    if not tasks:
        print("Nothing to do.")
        return

    est_cost = len(tasks) * 0.025
    print(f"Estimated cost: ~${est_cost:.2f}")

    counter = {"labeled": 0, "errors": 0, "transactions": 0, "total": len(tasks)}
    semaphore = asyncio.Semaphore(concurrency)

    coros = [
        process_page(client, semaphore, img, lbl, stem, counter)
        for img, lbl, stem in tasks
    ]

    await asyncio.gather(*coros)

    print(f"\nDone. Labeled {counter['labeled']} pages, {counter['transactions']} transactions, {counter['errors']} errors")


if __name__ == "__main__":
    asyncio.run(main())
