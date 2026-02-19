#!/usr/bin/env python3
"""
Terminal UI for reviewing and correcting BIO-aligned labels.

Shows page image path, word tokens with BIO tags, and Claude's transactions.
Accept, reject, or edit per page.

Usage:
    python review_labels.py                      # review all pages
    python review_labels.py --pdf app_123        # review specific PDF
    python review_labels.py --low-alignment 80   # only pages below 80% alignment
    python review_labels.py --stats              # show review statistics
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PAGES_DIR = PROJECT_DIR / "data" / "pages"
LABELS_DIR = PROJECT_DIR / "data" / "labels"
TRAINING_DIR = PROJECT_DIR / "data" / "training"

# Tag colors for terminal display
TAG_COLORS = {
    "B-DATE": "\033[94m",     # blue
    "I-DATE": "\033[94m",
    "B-DESC": "\033[92m",     # green
    "I-DESC": "\033[92m",
    "B-AMOUNT": "\033[93m",   # yellow
    "I-AMOUNT": "\033[93m",
    "B-BALANCE": "\033[95m",  # magenta
    "I-BALANCE": "\033[95m",
    "O": "\033[90m",          # gray
}
RESET = "\033[0m"


def display_page(bio_data: dict, label_data: dict):
    """Display a page's BIO tags in a readable format."""
    words = bio_data.get("words", [])
    tags = bio_data.get("tags", [])
    tag_counts = bio_data.get("tag_counts", {})
    alignment_rate = bio_data.get("alignment_rate", 0)

    print(f"\n{'='*80}")
    print(f"PDF: {bio_data['pdf_stem']}  Page: {bio_data['page_index']}")
    print(f"Words: {len(words)}  Alignment: {alignment_rate}%")
    print(f"Tags: {tag_counts}")
    print(f"{'='*80}")

    # Show Claude's transactions
    txns = label_data.get("transactions", [])
    if txns:
        print(f"\nClaude Vision found {len(txns)} transactions:")
        for j, t in enumerate(txns):
            print(f"  {j+1}. {t.get('date','?')} | {t.get('description','')[:50]} | "
                  f"${t.get('amount',0):,.2f} | bal: {t.get('balance_after','n/a')}")

    # Show tagged words (grouped by row)
    print(f"\nBIO-tagged tokens:")
    current_line = []
    prev_top = None

    for word, tag in zip(words, tags):
        color = TAG_COLORS.get(tag, "")
        tagged = f"{color}[{word}/{tag}]{RESET}"

        # Check for visual line break (would need bbox data for perfect grouping)
        current_line.append(tagged)

        if len(current_line) >= 12:
            print("  " + " ".join(current_line))
            current_line = []

    if current_line:
        print("  " + " ".join(current_line))

    # Show reconstructed transactions from BIO tags
    print(f"\nReconstructed from BIO tags:")
    current_txn = {}
    current_field = None
    txn_list = []

    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current_txn and "date" in current_txn:
                txn_list.append(current_txn.copy())
                if tag == "B-DATE":
                    current_txn = {}

            field = tag[2:].lower()
            current_txn[field] = word
            current_field = field
        elif tag.startswith("I-"):
            field = tag[2:].lower()
            if field in current_txn:
                current_txn[field] += f" {word}"
        elif tag == "O":
            current_field = None

    if current_txn and "date" in current_txn:
        txn_list.append(current_txn)

    for j, t in enumerate(txn_list):
        print(f"  {j+1}. date={t.get('date','?')} | desc={t.get('desc','')[:50]} | "
              f"amount={t.get('amount','?')} | bal={t.get('balance','n/a')}")


def review_page(pdf_stem: str, page_idx: int) -> str:
    """Review a single page. Returns 'accept', 'reject', or 'skip'."""
    bio_path = TRAINING_DIR / pdf_stem / f"page_{page_idx}_bio.json"
    label_path = LABELS_DIR / pdf_stem / f"page_{page_idx}_labels.json"

    if not bio_path.exists():
        return "skip"

    with open(bio_path) as f:
        bio_data = json.load(f)

    label_data = {}
    if label_path.exists():
        with open(label_path) as f:
            label_data = json.load(f)

    display_page(bio_data, label_data)

    print(f"\n[a]ccept  [r]eject  [s]kip  [q]uit")
    choice = input("> ").strip().lower()

    if choice == "a":
        bio_data["reviewed"] = True
        bio_data["review_status"] = "accepted"
        with open(bio_path, "w") as f:
            json.dump(bio_data, f, indent=2)
        return "accept"
    elif choice == "r":
        bio_data["reviewed"] = True
        bio_data["review_status"] = "rejected"
        reason = input("Rejection reason (optional): ").strip()
        if reason:
            bio_data["rejection_reason"] = reason
        with open(bio_path, "w") as f:
            json.dump(bio_data, f, indent=2)
        return "reject"
    elif choice == "q":
        return "quit"
    else:
        return "skip"


def show_stats():
    """Show review statistics across all training data."""
    if not TRAINING_DIR.exists():
        print("No training data. Run align_labels.py first.")
        return

    total = 0
    reviewed = 0
    accepted = 0
    rejected = 0
    alignment_rates = []

    for pdf_dir in sorted(TRAINING_DIR.iterdir()):
        if not pdf_dir.is_dir():
            continue
        for bio_file in sorted(pdf_dir.glob("page_*_bio.json")):
            with open(bio_file) as f:
                data = json.load(f)
            total += 1
            alignment_rates.append(data.get("alignment_rate", 0))
            if data.get("reviewed"):
                reviewed += 1
                if data.get("review_status") == "accepted":
                    accepted += 1
                elif data.get("review_status") == "rejected":
                    rejected += 1

    print(f"Review Statistics:")
    print(f"  Total pages:    {total}")
    print(f"  Reviewed:       {reviewed} ({reviewed/total*100:.0f}%)" if total else "")
    print(f"  Accepted:       {accepted}")
    print(f"  Rejected:       {rejected}")
    print(f"  Unreviewed:     {total - reviewed}")

    if alignment_rates:
        avg = sum(alignment_rates) / len(alignment_rates)
        below_90 = sum(1 for r in alignment_rates if r < 90)
        print(f"\n  Avg alignment:  {avg:.1f}%")
        print(f"  Below 90%:      {below_90}")


def main():
    parser = argparse.ArgumentParser(description="Review BIO-aligned labels")
    parser.add_argument("--pdf", help="Review specific PDF stem")
    parser.add_argument("--low-alignment", type=float, help="Only review pages below this alignment %")
    parser.add_argument("--unreviewed", action="store_true", help="Only show unreviewed pages")
    parser.add_argument("--stats", action="store_true", help="Show review statistics")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if not TRAINING_DIR.exists():
        print("No training data. Run align_labels.py first.")
        sys.exit(1)

    # Collect pages to review
    pages_to_review = []

    if args.pdf:
        pdf_dirs = [TRAINING_DIR / args.pdf]
    else:
        pdf_dirs = sorted([d for d in TRAINING_DIR.iterdir() if d.is_dir()])

    for pdf_dir in pdf_dirs:
        for bio_file in sorted(pdf_dir.glob("page_*_bio.json")):
            with open(bio_file) as f:
                data = json.load(f)

            if args.unreviewed and data.get("reviewed"):
                continue

            if args.low_alignment and data.get("alignment_rate", 100) >= args.low_alignment:
                continue

            page_idx = int(bio_file.stem.replace("page_", "").replace("_bio", ""))
            pages_to_review.append((pdf_dir.name, page_idx))

    if not pages_to_review:
        print("No pages to review matching criteria.")
        return

    print(f"Found {len(pages_to_review)} pages to review")

    accepted = 0
    rejected = 0
    skipped = 0

    for i, (stem, page_idx) in enumerate(pages_to_review):
        print(f"\n[{i+1}/{len(pages_to_review)}]")
        result = review_page(stem, page_idx)

        if result == "accept":
            accepted += 1
        elif result == "reject":
            rejected += 1
        elif result == "quit":
            break
        else:
            skipped += 1

    print(f"\nSession: {accepted} accepted, {rejected} rejected, {skipped} skipped")


if __name__ == "__main__":
    main()
