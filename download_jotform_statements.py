#!/usr/bin/env python3
"""
Download bank statement PDFs from Jotform submissions.

Connects to the Jotform API, lists all submissions with file uploads,
downloads the PDFs, and saves them to data/pdfs/ for the training pipeline.

Setup:
    1. Get your API key from: Jotform Settings > API > Create New Key
    2. Add to .env: JOTFORM_API_KEY=your_key_here
    3. Run: python download_jotform_statements.py

Pipeline after download:
    python pdf_to_pages.py          # convert new PDFs to page images
    python label_rules.py --force   # auto-label with BIO tags
    python train.py                 # retrain model

Usage:
    python download_jotform_statements.py                # download all
    python download_jotform_statements.py --dry-run      # list without downloading
    python download_jotform_statements.py --form FORM_ID # specific form only
    python download_jotform_statements.py --limit 50     # max submissions to process
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

PDF_DIR = PROJECT_DIR / "data" / "pdfs"
JOTFORM_API_BASE = "https://api.jotform.com"


def get_api_key() -> str:
    key = os.environ.get("JOTFORM_API_KEY", "").strip()
    if not key:
        print("Error: JOTFORM_API_KEY not set in .env")
        print("Get your API key from: Jotform Settings > API > Create New Key")
        print("Then add to .env: JOTFORM_API_KEY=your_key_here")
        sys.exit(1)
    return key


def api_get(endpoint: str, api_key: str, params: dict = None) -> dict:
    """Make a GET request to the Jotform API."""
    url = f"{JOTFORM_API_BASE}{endpoint}"
    headers = {"APIKEY": api_key}
    resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_forms(api_key: str) -> list:
    """List all forms in the account."""
    forms = []
    offset = 0
    limit = 100

    while True:
        data = api_get("/user/forms", api_key, {"offset": offset, "limit": limit})
        content = data.get("content", [])
        if not content:
            break
        forms.extend(content)
        offset += limit
        if len(content) < limit:
            break

    return forms


def list_submissions(api_key: str, form_id: str, limit: int = 1000) -> list:
    """List all submissions for a form."""
    submissions = []
    offset = 0
    batch = 100

    while len(submissions) < limit:
        fetch = min(batch, limit - len(submissions))
        data = api_get(
            f"/form/{form_id}/submissions",
            api_key,
            {"offset": offset, "limit": fetch},
        )
        content = data.get("content", [])
        if not content:
            break
        submissions.extend(content)
        offset += fetch
        if len(content) < fetch:
            break

    return submissions[:limit]


def extract_file_urls(submission: dict) -> list:
    """Extract all file upload URLs from a submission's answers."""
    urls = []
    answers = submission.get("answers", {})

    for field_id, field_data in answers.items():
        # File upload fields have type "control_fileupload"
        if field_data.get("type") in ("control_fileupload", "control_widget"):
            answer = field_data.get("answer", [])
            if isinstance(answer, list):
                for url in answer:
                    if isinstance(url, str) and url.lower().endswith(".pdf"):
                        urls.append(url)
            elif isinstance(answer, str) and answer.lower().endswith(".pdf"):
                urls.append(answer)

        # Also check prettyFormat which sometimes has the URL
        pretty = field_data.get("prettyFormat", "")
        if isinstance(pretty, str) and ".pdf" in pretty.lower():
            # Extract URLs from HTML links
            pdf_urls = re.findall(r'href=["\']?(https?://[^"\'>\s]+\.pdf)', pretty, re.I)
            urls.extend(pdf_urls)

        # Check for answer as a string URL
        answer_str = field_data.get("answer", "")
        if isinstance(answer_str, str) and answer_str.startswith("http") and ".pdf" in answer_str.lower():
            urls.append(answer_str)

    return list(set(urls))  # deduplicate


def download_pdf(url: str, dest_path: Path, api_key: str) -> bool:
    """Download a PDF file from Jotform."""
    try:
        headers = {"APIKEY": api_key}
        resp = requests.get(url, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()

        # Verify it's actually a PDF
        content_type = resp.headers.get("content-type", "")
        first_bytes = b""

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not first_bytes:
                    first_bytes = chunk[:5]
                f.write(chunk)

        # Check PDF magic bytes
        if not first_bytes.startswith(b"%PDF"):
            dest_path.unlink(missing_ok=True)
            return False

        # Verify it looks like a bank statement (not a voided check, ID, etc.)
        if not _verify_bank_statement(dest_path):
            dest_path.unlink(missing_ok=True)
            return False

        return True

    except Exception:
        dest_path.unlink(missing_ok=True)
        return False


def _verify_bank_statement(pdf_path: Path) -> bool:
    """Open the PDF and verify it actually looks like a bank statement.

    Checks for bank statement indicators in the text:
    - Bank names or FDIC references
    - Account numbers
    - Statement period / date ranges
    - Transaction-related keywords (deposit, withdrawal, balance)
    - Dollar amounts in table format

    Returns False for voided checks, IDs, applications, etc.
    """
    try:
        import pdfplumber
        pdf = pdfplumber.open(str(pdf_path))

        if len(pdf.pages) == 0:
            pdf.close()
            return False

        # Extract text from first 2 pages
        text = ""
        for p in pdf.pages[:2]:
            page_text = p.extract_text() or ""
            text += page_text + "\n"
        pdf.close()

        text_lower = text.lower()

        # Too little text — likely a scanned image or non-document
        if len(text_lower.strip()) < 100:
            return False

        # Must have at least 2 of these bank statement indicators
        indicators = 0

        # Bank / financial institution markers
        bank_kw = ["bank", "credit union", "member fdic", "fdic", "financial",
                    "n.a.", "national association"]
        if any(kw in text_lower for kw in bank_kw):
            indicators += 1

        # Account reference
        acct_kw = ["account", "acct", "account number", "account #"]
        if any(kw in text_lower for kw in acct_kw):
            indicators += 1

        # Statement period
        period_kw = ["statement period", "statement date", "statement for",
                     "period ending", "through", "beginning balance",
                     "ending balance", "opening balance", "closing balance"]
        if any(kw in text_lower for kw in period_kw):
            indicators += 1

        # Transaction keywords
        txn_kw = ["deposit", "withdrawal", "debit", "credit", "transaction",
                  "check", "transfer", "payment", "balance"]
        txn_matches = sum(1 for kw in txn_kw if kw in text_lower)
        if txn_matches >= 3:
            indicators += 1

        # Dollar amounts (common in statements)
        import re
        dollar_pattern = re.findall(r'\$[\d,]+\.\d{2}', text)
        if len(dollar_pattern) >= 3:
            indicators += 1

        # Need at least 2 indicators to qualify as a bank statement
        return indicators >= 2

    except Exception:
        return False


def is_bank_statement(url: str) -> bool:
    """Filter out non-bank-statement files by filename patterns."""
    name = Path(urlparse(url).path).name.lower()

    # Skip these file types entirely
    SKIP_PATTERNS = [
        # Identity documents
        "driver", "license", "passport", "id_", "id.", "dl_", "dl.",
        "ssn", "social_security", "ein",
        # Checks
        "void", "voided", "check", "cheque", "canceled_check", "cancelled",
        # Applications / forms
        "application", "agreement", "contract", "authorization",
        "w-2", "w2", "w-9", "w9", "1099", "1098", "tax_return", "schedule",
        # Invoices / receipts
        "invoice", "receipt", "bill_of", "purchase_order", "po_",
        # Other non-statements
        "certificate", "letter", "notice", "memo",
        "ach_", "wire_transfer", "wire_instruction",
        "pay_stub", "paystub", "payslip", "earnings",
        "balance_sheet", "profit_loss", "p&l", "pnl",
        "articles", "incorporation", "bylaws", "resolution",
        "lease", "rental", "insurance", "policy",
        "photo", "selfie", "headshot", "scan",
        "img", "image", "screenshot",
        "direct_deposit", "deposit_form", "deposit-form",
        "front_back", "front_and_back",
    ]

    for pattern in SKIP_PATTERNS:
        # Match with common separators: space, underscore, hyphen, %20
        normalized = name.replace("%20", "_").replace("-", "_").replace(" ", "_")
        if pattern.replace("_", "") in normalized.replace("_", ""):
            return False

    # Must end with .pdf
    if not name.endswith(".pdf"):
        return False

    # Skip very short filenames (likely not statements)
    base = name.replace(".pdf", "")
    if len(base) < 3:
        return False

    return True


def safe_filename(url: str, submission_id: str) -> str:
    """Generate a safe filename from a URL and submission ID."""
    parsed = urlparse(url)
    original_name = Path(parsed.path).name
    # Clean the filename
    clean = re.sub(r'[^\w\-.]', '_', original_name)
    # Prefix with submission ID to avoid collisions
    return f"jf_{submission_id}_{clean}"


def main():
    parser = argparse.ArgumentParser(description="Download bank statements from Jotform")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    parser.add_argument("--form", help="Specific form ID to process")
    parser.add_argument("--limit", type=int, default=1000, help="Max submissions per form")
    parser.add_argument("--list-forms", action="store_true", help="Just list all forms and exit")
    args = parser.parse_args()

    api_key = get_api_key()

    # List forms
    print("Fetching forms...")
    forms = list_forms(api_key)
    print(f"Found {len(forms)} forms\n")

    if args.list_forms:
        print(f"{'Form ID':<15} {'Submissions':>12} {'Title'}")
        print("-" * 70)
        for form in forms:
            print(f"{form['id']:<15} {form.get('count', '?'):>12} {form.get('title', 'Untitled')}")
        return

    # Filter to specific form if requested
    if args.form:
        forms = [f for f in forms if f["id"] == args.form]
        if not forms:
            print(f"Form {args.form} not found")
            sys.exit(1)

    # Get existing PDFs to skip duplicates
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in PDF_DIR.glob("*.pdf")}
    print(f"Existing training PDFs: {len(existing)}")

    # Process each form
    total_found = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    total_filtered = 0  # non-statement files skipped

    for form in forms:
        form_id = form["id"]
        form_title = form.get("title", "Untitled")
        submission_count = int(form.get("count", 0))

        if submission_count == 0:
            continue

        print(f"\n{'='*60}")
        print(f"Form: {form_title} (ID: {form_id}, {submission_count} submissions)")
        print(f"{'='*60}")

        submissions = list_submissions(api_key, form_id, limit=args.limit)
        print(f"  Fetched {len(submissions)} submissions")

        form_pdfs = 0

        for sub in submissions:
            sub_id = sub.get("id", "unknown")
            pdf_urls = extract_file_urls(sub)

            for url in pdf_urls:
                total_found += 1

                # Skip non-bank-statement files
                if not is_bank_statement(url):
                    total_filtered += 1
                    continue

                filename = safe_filename(url, sub_id)

                if filename in existing:
                    total_skipped += 1
                    continue

                if args.dry_run:
                    print(f"  [DRY] {filename}")
                    form_pdfs += 1
                    continue

                dest = PDF_DIR / filename
                print(f"  Downloading: {filename}...", end=" ", flush=True)

                if download_pdf(url, dest, api_key):
                    size_kb = dest.stat().st_size / 1024
                    print(f"OK ({size_kb:.0f} KB)")
                    total_downloaded += 1
                    form_pdfs += 1
                    existing.add(filename)
                else:
                    print("SKIPPED (not a valid PDF or not a bank statement)")
                    total_failed += 1

                # Rate limit: ~5 requests/sec (Gold plan: 100K/day)
                time.sleep(0.2)

        print(f"  PDFs from this form: {form_pdfs}")

    # Summary
    statements_found = total_found - total_filtered
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Total PDF files:    {total_found}")
    print(f"  Non-statements:     {total_filtered} (voided checks, IDs, apps, etc.)")
    print(f"  Bank statements:    {statements_found}")
    print(f"  Already had:        {total_skipped}")
    if args.dry_run:
        print(f"  Would download:     {statements_found - total_skipped}")
        print(f"\n  (Dry run — remove --dry-run to download)")
    else:
        print(f"  Downloaded:         {total_downloaded}")
        print(f"  Failed:             {total_failed}")

    if total_downloaded > 0:
        print(f"\nNext steps:")
        print(f"  1. python pdf_to_pages.py          # Convert new PDFs to page images")
        print(f"  2. python label_rules.py --force   # Auto-label with BIO tags")
        print(f"  3. python train.py                 # Retrain model")


if __name__ == "__main__":
    main()
