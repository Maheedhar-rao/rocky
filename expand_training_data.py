#!/usr/bin/env python3
"""
Expand training data with diverse bank statement formats.

Scans existing PDF collections on disk, identifies distinct bank formats,
selects a diverse sample (2-3 PDFs per bank), and copies them to data/pdfs/
for processing through the training pipeline.

Pipeline after this script:
    1. python pdf_to_pages.py          # convert new PDFs to page images + words
    2. python label_rules.py --force   # auto-label with BIO tags (free, rule-based)
    3. python train.py                 # retrain model with expanded dataset

Usage:
    python expand_training_data.py                    # scan + select + copy
    python expand_training_data.py --dry-run          # scan + show what would be copied
    python expand_training_data.py --max-per-bank 5   # up to 5 PDFs per bank
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pdfplumber

PROJECT_DIR = Path(__file__).resolve().parent
PDF_DIR = PROJECT_DIR / "data" / "pdfs"

# Directories to scan for bank statement PDFs
SCAN_DIRS = [
    Path.home() / "Desktop" / "bank statements",
    Path.home() / "Downloads",
]

# Banks already well-represented in training data — we'll still include a few
# but prioritize banks NOT in this list
EXISTING_BANKS = {"chase", "wells fargo", "pnc", "twidcorp"}

# Known bank name patterns — order matters (check specific before generic)
BANK_PATTERNS = [
    (r"bank\s*of\s*america|bofa", "Bank of America"),
    (r"chase|jpmorgan", "Chase"),
    (r"wells\s*fargo", "Wells Fargo"),
    (r"td\s*bank", "TD Bank"),
    (r"us\s*bank|u\.s\.\s*bank", "US Bank"),
    (r"pnc\s*bank|pnc\b", "PNC Bank"),
    (r"capital\s*one", "Capital One"),
    (r"citibank|citi\b", "Citibank"),
    (r"navy\s*federal", "Navy Federal CU"),
    (r"huntington", "Huntington Bank"),
    (r"regions\s*bank|regions\b", "Regions Bank"),
    (r"truist|bb&t|suntrust", "Truist"),
    (r"bmo\b|bmo\s*harris", "BMO"),
    (r"citizens\s*bank|citizens\b", "Citizens Bank"),
    (r"keybank|key\s*bank", "KeyBank"),
    (r"zions\s*bank|zions\b", "Zions Bank"),
    (r"comerica", "Comerica"),
    (r"m&t\s*bank", "M&T Bank"),
    (r"fifth\s*third", "Fifth Third"),
    (r"synovus", "Synovus"),
    (r"first\s*horizon", "First Horizon"),
    (r"bluevine", "Bluevine"),
    (r"mercury", "Mercury"),
    (r"brex\b", "Brex"),
    (r"square|block\b", "Square/Block"),
    (r"paypal", "PayPal"),
    (r"american\s*express|amex", "American Express"),
    (r"discover\s*bank|discover\b", "Discover"),
    (r"ally\s*bank|ally\b", "Ally Bank"),
    (r"webster\s*bank", "Webster Bank"),
    (r"fulton\s*bank", "Fulton Bank"),
    (r"banner\s*bank", "Banner Bank"),
    (r"south\s*state", "South State Bank"),
    (r"valley\s*national", "Valley National Bank"),
    (r"prosperity\s*bank", "Prosperity Bank"),
    (r"city\s*national", "City National Bank"),
    (r"f&m\s*bank|farmers\s*&\s*merchants", "F&M Bank"),
    (r"central\s*bank", "Central Bank"),
    (r"community\s*bank", "Community Bank"),
    (r"first\s*national", "First National Bank"),
    (r"twidcorp", "TWIDCORP/PNC"),
    (r"yellowstone", "Yellowstone Bank"),
    (r"finemark", "Finemark National Bank"),
    (r"south\s*shore", "South Shore Bank"),
    (r"diamond\s*bank", "Diamond Bank"),
    (r"ameristate", "AmeriState Bank"),
    # Generic credit union detection
    (r"credit\s*union|federal\s*credit", "Credit Union"),
    # Fintech / neobank detection
    (r"relay|novo|lili|found\b|grasshopper", "Fintech/Neobank"),
]


def detect_bank_from_text(text: str) -> str:
    """Detect bank name from extracted PDF text."""
    if not text:
        return "Unknown"

    text_lower = text[:3000].lower()  # Only check first ~3000 chars (header area)

    for pattern, bank_name in BANK_PATTERNS:
        if re.search(pattern, text_lower):
            return bank_name

    # Look for "Member FDIC" nearby text for unknown banks
    if "member fdic" in text_lower or "fdic" in text_lower:
        # Try to find bank name near top of document
        lines = text[:1000].split("\n")
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 3 and "bank" in line.lower():
                return line[:50]  # Use first 50 chars as bank name

    return "Unknown"


def scan_pdfs(scan_dirs: list, existing_pdfs: set) -> dict:
    """Scan directories for PDF files and group by bank.

    Returns: {bank_name: [(path, page_count), ...]}
    """
    bank_pdfs = defaultdict(list)
    total_scanned = 0
    errors = 0

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            print(f"  Skipping {scan_dir} (not found)")
            continue

        pdf_files = list(scan_dir.rglob("*.pdf"))
        print(f"  Found {len(pdf_files)} PDFs in {scan_dir}")

        for pdf_path in pdf_files:
            total_scanned += 1

            # Skip if already in training data
            if pdf_path.name in existing_pdfs:
                continue

            # Skip very small files (likely not statements)
            if pdf_path.stat().st_size < 10_000:
                continue

            # Skip files that are clearly not bank statements
            name_lower = pdf_path.name.lower()
            skip_keywords = [
                "application", "tax", "license", "contract", "agreement",
                "invoice", "receipt", "letter", "certificate", "id_",
                "passport", "w-2", "w2", "1099", "schedule", "form",
            ]
            if any(kw in name_lower for kw in skip_keywords):
                continue

            # Extract text and detect bank
            try:
                pdf = pdfplumber.open(str(pdf_path))
                page_count = len(pdf.pages)
                text = ""
                for p in pdf.pages[:2]:  # Only first 2 pages for detection
                    page_text = p.extract_text() or ""
                    text += page_text + "\n"
                pdf.close()

                if len(text.strip()) < 50:
                    continue  # Likely scanned/image PDF — skip for now

                bank = detect_bank_from_text(text)
                bank_pdfs[bank].append((pdf_path, page_count))

            except Exception:
                errors += 1
                continue

            if total_scanned % 100 == 0:
                print(f"    Scanned {total_scanned} PDFs...")

    print(f"  Total scanned: {total_scanned}, Errors: {errors}")
    return dict(bank_pdfs)


def select_diverse_sample(bank_pdfs: dict, max_per_bank: int = 3) -> list:
    """Select a diverse sample of PDFs across all bank formats.

    Prioritizes:
    1. Banks NOT already in training data
    2. PDFs with 2-8 pages (good training size)
    3. Variety within each bank (different page counts)

    Returns: list of (path, bank_name) tuples
    """
    selected = []

    # Sort banks: new banks first, then existing banks
    banks_sorted = sorted(
        bank_pdfs.keys(),
        key=lambda b: (b.lower() in EXISTING_BANKS or b == "Unknown", b),
    )

    for bank in banks_sorted:
        pdfs = bank_pdfs[bank]

        if bank == "Unknown":
            # For unknown banks, take up to max_per_bank
            # These are likely diverse community banks
            # Prefer ones with reasonable page counts
            pdfs_sorted = sorted(pdfs, key=lambda x: abs(x[1] - 4))  # prefer ~4 pages
            for path, pages in pdfs_sorted[:max_per_bank]:
                selected.append((path, bank, pages))
            continue

        if bank.lower() in EXISTING_BANKS:
            # Already have this bank — take just 1 more for variety
            limit = 1
        else:
            limit = max_per_bank

        # Sort by page count variety — pick different sizes
        pdfs_sorted = sorted(pdfs, key=lambda x: x[1])

        # Pick evenly spaced by page count
        if len(pdfs_sorted) <= limit:
            chosen = pdfs_sorted
        else:
            step = len(pdfs_sorted) / limit
            chosen = [pdfs_sorted[int(i * step)] for i in range(limit)]

        for path, pages in chosen:
            selected.append((path, bank, pages))

    return selected


def main():
    parser = argparse.ArgumentParser(description="Expand training data with diverse bank formats")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without copying files")
    parser.add_argument("--max-per-bank", type=int, default=3, help="Max PDFs per bank format")
    parser.add_argument("--include-unknown", action="store_true", help="Include unidentified bank PDFs")
    args = parser.parse_args()

    # Get existing training PDFs
    existing_pdfs = set()
    if PDF_DIR.exists():
        existing_pdfs = {p.name for p in PDF_DIR.glob("*.pdf")}
    print(f"Existing training PDFs: {len(existing_pdfs)}")

    # Scan for new PDFs
    print(f"\nScanning for bank statement PDFs...")
    bank_pdfs = scan_pdfs(SCAN_DIRS, existing_pdfs)

    if not args.include_unknown and "Unknown" in bank_pdfs:
        unknown_count = len(bank_pdfs["Unknown"])
        del bank_pdfs["Unknown"]
        print(f"  (Skipping {unknown_count} unidentified PDFs. Use --include-unknown to include)")

    # Show what we found
    print(f"\nDistinct bank formats found: {len(bank_pdfs)}")
    total_pdfs = sum(len(v) for v in bank_pdfs.values())
    print(f"Total statement PDFs: {total_pdfs}")

    print(f"\n{'Bank':<30} {'PDFs':>6} {'Already Training':>18}")
    print("-" * 58)
    for bank in sorted(bank_pdfs.keys()):
        count = len(bank_pdfs[bank])
        in_training = "YES" if bank.lower() in EXISTING_BANKS else ""
        print(f"{bank:<30} {count:>6} {in_training:>18}")

    # Select diverse sample
    selected = select_diverse_sample(bank_pdfs, max_per_bank=args.max_per_bank)

    new_banks = set()
    new_pages = 0
    for path, bank, pages in selected:
        if bank.lower() not in EXISTING_BANKS:
            new_banks.add(bank)
        new_pages += pages

    print(f"\n{'='*58}")
    print(f"Selected: {len(selected)} PDFs (~{new_pages} pages) from {len(new_banks)} NEW bank formats")
    print(f"\nPDFs to add:")
    for path, bank, pages in sorted(selected, key=lambda x: x[1]):
        print(f"  [{bank:<25}] {pages:>3}pp  {path.name}")

    if args.dry_run:
        print(f"\n(Dry run — no files copied. Remove --dry-run to proceed)")
        return

    # Copy selected PDFs to data/pdfs/
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path, bank, pages in selected:
        dest = PDF_DIR / path.name
        # Handle duplicate filenames by prefixing with bank name
        if dest.exists():
            safe_bank = re.sub(r'[^\w]', '_', bank)[:20]
            dest = PDF_DIR / f"{safe_bank}_{path.name}"

        if not dest.exists():
            shutil.copy2(str(path), str(dest))
            copied += 1

    print(f"\nCopied {copied} PDFs to {PDF_DIR}")
    print(f"\nNext steps:")
    print(f"  1. python pdf_to_pages.py          # Convert new PDFs to page images")
    print(f"  2. python label_rules.py --force   # Auto-label with BIO tags")
    print(f"  3. python train.py                 # Retrain model")


if __name__ == "__main__":
    main()
