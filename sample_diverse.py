#!/usr/bin/env python3
"""
Select a diverse sample of bank statement PDFs for Claude labeling.

Picks PDFs across bank formats to maximise model training diversity.
Skips PDFs that already have Claude labels. Outputs a manifest file
listing the selected PDF stems.

Usage:
    python sample_diverse.py                    # select ~60 diverse PDFs
    python sample_diverse.py --target-pages 800 # aim for ~800 pages
    python sample_diverse.py --dry-run          # just show the plan
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pdfplumber

PROJECT_DIR = Path(__file__).resolve().parent
PDF_DIR = PROJECT_DIR / "data" / "pdfs"
PAGES_DIR = PROJECT_DIR / "data" / "pages"
LABELS_DIR = PROJECT_DIR / "data" / "labels"

# Bank detection patterns: (name, [text_patterns])
BANKS = [
    ("Chase", [r"(?i)jpmorgan\s*chase", r"(?i)chase\b.*bank", r"(?i)\bchase\b"]),
    ("Wells Fargo", [r"(?i)wells\s*fargo"]),
    ("Bank of America", [r"(?i)bank\s*of\s*america"]),
    ("Truist", [r"(?i)truist"]),
    ("Navy Federal", [r"(?i)navy\s*federal"]),
    ("PNC", [r"(?i)\bpnc\s*(bank|financial)?"]),
    ("US Bank", [r"(?i)u\.?s\.?\s*bank", r"(?i)us\s*bancorp"]),
    ("TD Bank", [r"(?i)td\s*bank"]),
    ("Capital One", [r"(?i)capital\s*one"]),
    ("Regions", [r"(?i)regions\s*(bank|financial)"]),
    ("Zions / NBAZ", [r"(?i)zions", r"(?i)national\s*bank\s*of\s*arizona"]),
    ("Huntington", [r"(?i)huntington"]),
    ("Citizens", [r"(?i)citizens\s*(bank|financial)"]),
    ("USAA", [r"(?i)\busaa\b"]),
    ("BMO / Harris", [r"(?i)\bbmo\b", r"(?i)harris\s*bank"]),
    ("KeyBank", [r"(?i)key\s*bank", r"(?i)keycorp"]),
    ("First Horizon", [r"(?i)first\s*horizon"]),
    ("Fifth Third", [r"(?i)fifth\s*third"]),
    ("Citibank", [r"(?i)citi\s*bank", r"(?i)citigroup"]),
    ("Ally", [r"(?i)ally\s*(bank|financial)"]),
    ("Comerica", [r"(?i)comerica"]),
    ("Credit Union", [r"(?i)credit\s*union"]),
    ("Community Bank", [r"(?i)community\s*bank", r"(?i)savings\s*bank"]),
]

# Per-bank target PDFs (more for common banks, fewer for rare)
BANK_TARGETS = {
    "Chase": 6, "Wells Fargo": 8, "Bank of America": 8,
    "Truist": 5, "Navy Federal": 5, "PNC": 5,
    "US Bank": 3, "TD Bank": 4, "Capital One": 4,
    "Regions": 4, "Zions / NBAZ": 3, "Huntington": 2,
    "Citizens": 2, "USAA": 2, "BMO / Harris": 2,
    "KeyBank": 2, "First Horizon": 2, "Fifth Third": 2,
    "Citibank": 2, "Ally": 2, "Comerica": 2,
    "Credit Union": 4, "Community Bank": 3,
}


def detect_bank(text: str) -> str:
    """Detect bank from first-page text. Returns bank name or 'Unknown'."""
    for name, patterns in BANKS:
        for pat in patterns:
            if re.search(pat, text):
                return name
    return "Unknown"


def count_pages(pdf_path: Path) -> int:
    """Count pages in a PDF without full extraction."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


def get_first_page_text(pdf_path: Path) -> str:
    """Extract first page text."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if pdf.pages:
                return pdf.pages[0].extract_text() or ""
    except Exception:
        pass
    return ""


def main():
    parser = argparse.ArgumentParser(description="Select diverse PDFs for labeling")
    parser.add_argument("--target-pages", type=int, default=500,
                        help="Target total pages to label (default: 500)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without writing manifest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Get all PDFs
    all_pdfs = sorted(PDF_DIR.glob("*.[pP][dD][fF]"))
    print(f"Total PDFs: {len(all_pdfs)}")

    # Get already-labeled PDFs (Claude labels)
    labeled_stems = set()
    if LABELS_DIR.exists():
        labeled_stems = {d.name for d in LABELS_DIR.iterdir() if d.is_dir()}
    print(f"Already Claude-labeled: {len(labeled_stems)} PDFs")

    # Classify all PDFs by bank (scan first page text)
    print("\nClassifying PDFs by bank (scanning first pages)...")
    bank_to_pdfs = defaultdict(list)
    errors = 0

    for i, pdf_path in enumerate(all_pdfs):
        if pdf_path.stem in labeled_stems:
            continue  # Skip already labeled

        text = get_first_page_text(pdf_path)
        if not text:
            errors += 1
            continue

        bank = detect_bank(text)
        bank_to_pdfs[bank].append(pdf_path)

        if (i + 1) % 500 == 0:
            print(f"  Scanned {i+1}/{len(all_pdfs)}...")

    print(f"\nBank distribution (unlabeled PDFs):")
    for bank, pdfs in sorted(bank_to_pdfs.items(), key=lambda x: -len(x[1])):
        print(f"  {bank:<25s} {len(pdfs):>5d} PDFs")
    if errors:
        print(f"  (skipped {errors} unreadable PDFs)")

    # Select diverse sample
    selected = []
    total_pages_est = 0

    print(f"\nSelecting diverse sample (target: ~{args.target_pages} pages)...")

    for bank in sorted(BANK_TARGETS.keys(), key=lambda b: -BANK_TARGETS.get(b, 2)):
        available = bank_to_pdfs.get(bank, [])
        target = BANK_TARGETS.get(bank, 2)

        if not available:
            print(f"  {bank:<25s} 0 available — skipping")
            continue

        # Shuffle and pick
        random.shuffle(available)
        picked = available[:target]
        for p in picked:
            pages = count_pages(p)
            selected.append({"path": str(p), "stem": p.stem, "bank": bank, "pages": pages})
            total_pages_est += pages

        print(f"  {bank:<25s} {len(picked):>2d}/{target} picked ({sum(s['pages'] for s in selected if s['bank'] == bank)} pages)")

    # Also pick some "Unknown" bank PDFs for long-tail coverage
    unknown = bank_to_pdfs.get("Unknown", [])
    if unknown:
        random.shuffle(unknown)
        # Pick enough to get close to target pages
        for p in unknown:
            if total_pages_est >= args.target_pages:
                break
            pages = count_pages(p)
            if pages > 0:
                selected.append({"path": str(p), "stem": p.stem, "bank": "Unknown", "pages": pages})
                total_pages_est += pages
        unk_count = sum(1 for s in selected if s["bank"] == "Unknown")
        print(f"  {'Unknown (long tail)':<25s} {unk_count:>2d} picked ({sum(s['pages'] for s in selected if s['bank'] == 'Unknown')} pages)")

    # Summary
    total_pdfs = len(selected)
    bank_counts = defaultdict(int)
    for s in selected:
        bank_counts[s["bank"]] += 1

    print(f"\n{'='*60}")
    print(f"Selected: {total_pdfs} PDFs, ~{total_pages_est} pages")
    print(f"Estimated Claude API cost: ${total_pages_est * 0.025:.2f}")
    print(f"Banks covered: {len(bank_counts)}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nDry run — no manifest written.")
        return

    # Write manifest
    manifest_path = PROJECT_DIR / "data" / "diverse_sample_manifest.json"
    manifest = {
        "total_pdfs": total_pdfs,
        "total_pages_est": total_pages_est,
        "banks_covered": len(bank_counts),
        "bank_counts": dict(bank_counts),
        "pdfs": selected,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")
    print(f"\nNext steps:")
    print(f"  1. python pdf_to_pages.py   (extract pages for new PDFs)")
    print(f"  2. python label_with_claude.py  (label with Claude Vision)")
    print(f"  3. python align_labels.py   (generate BIO training data)")
    print(f"  4. python train.py --claude-only  (retrain model)")


if __name__ == "__main__":
    main()
