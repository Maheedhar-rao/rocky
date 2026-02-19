#!/usr/bin/env python3
"""
Rule-based BIO tagger for bank statement pages.

Directly generates BIO tags from word tokens + bounding boxes,
without requiring Claude API calls. Uses regex patterns for dates/amounts
and positional heuristics for column detection.

Usage:
    python label_rules.py                    # label pages without existing BIO data
    python label_rules.py --force            # overwrite all (including Claude-aligned)
    python label_rules.py --pdf "name"       # label specific PDF
    python label_rules.py --dry-run          # count pages, don't write
    python label_rules.py --stats            # show tag distribution
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PAGES_DIR = PROJECT_DIR / "data" / "pages"
TRAINING_DIR = PROJECT_DIR / "data" / "training"

ROW_Y_TOLERANCE = 8  # same as align_labels.py

# --- Regex patterns (aligned with Statements_extractor.py) ---

DATE_RE = re.compile(
    r"^\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?$"
)

MONTH_NAMES = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april",
    "june", "july", "august", "september",
    "october", "november", "december",
}

AMOUNT_RE = re.compile(
    r"^\$?[-\(]?\d{1,3}(?:,\d{3})*\.\d{2}[-\)]?$"
)

# Section markers — skip these rows entirely
SECTION_WORDS = {
    "deposits", "additions", "withdrawals", "subtractions",
    "checks", "electronic", "other", "fees", "interest",
    "total", "totals", "subtotal", "continued", "summary",
    "daily", "ending", "beginning", "opening", "closing",
    "balance", "page", "statement",
}

SECTION_PHRASES = [
    "deposits and additions", "checks paid",
    "atm & debit card", "electronic withdrawals",
    "other withdrawals", "daily ending balance",
    "statement period", "account summary",
    "beginning balance", "ending balance",
    "total deposits", "total withdrawals",
    "total checks", "total fees",
]

HEADER_KEYWORDS = {
    "date": {"date", "posted", "post", "effective"},
    "description": {"description", "details", "transaction", "activity"},
    "amount": {"amount", "debit", "credit", "deposits", "withdrawals"},
    "balance": {"balance", "ending"},
}


def _group_words_by_row(words: list) -> list:
    """Group words into visual rows based on Y coordinate (top field)."""
    if not words:
        return []

    sorted_words = sorted(enumerate(words), key=lambda x: (x[1]["top"], x[1]["bbox"][0]))

    rows = []
    current_row = [sorted_words[0]]
    current_y = sorted_words[0][1]["top"]

    for idx_word in sorted_words[1:]:
        word_y = idx_word[1]["top"]
        if abs(word_y - current_y) <= ROW_Y_TOLERANCE:
            current_row.append(idx_word)
        else:
            current_row.sort(key=lambda x: x[1]["bbox"][0])
            rows.append(current_row)
            current_row = [idx_word]
            current_y = word_y

    if current_row:
        current_row.sort(key=lambda x: x[1]["bbox"][0])
        rows.append(current_row)

    return rows


def _is_date_token(text: str) -> bool:
    """Check if token matches a date pattern like MM/DD, M/D, MM-DD-YYYY."""
    return bool(DATE_RE.match(text))


def _is_month_name(text: str) -> bool:
    """Check if token is a month name (Jan, February, etc.)."""
    return text.lower().rstrip(".") in MONTH_NAMES


def _is_amount_token(text: str) -> bool:
    """Check if token matches a dollar amount like $1,234.56 or 500.00."""
    return bool(AMOUNT_RE.match(text))


def _is_section_marker(row: list) -> bool:
    """Check if a row is a section header, not a transaction."""
    row_text = " ".join(w["text"] for _, w in row).lower()

    for phrase in SECTION_PHRASES:
        if phrase in row_text:
            return True

    # Single-word section markers (only if row is short)
    if len(row) <= 3:
        first_word = row[0][1]["text"].lower().strip("*")
        if first_word in SECTION_WORDS:
            return True

    return False


def _is_header_row(row: list) -> bool:
    """Check if row contains column headers like DATE DESCRIPTION AMOUNT."""
    texts = {w["text"].lower() for _, w in row}
    # Need at least 2 header keywords from different categories
    cats_found = 0
    for cat_keywords in HEADER_KEYWORDS.values():
        if texts & cat_keywords:
            cats_found += 1
    return cats_found >= 2


def _detect_columns(rows: list) -> dict:
    """Detect column X positions from header rows. Returns fallback zones if none found."""
    columns = {
        "date_x": 0,
        "desc_x": 100,
        "amount_x": 650,
        "balance_x": 800,
    }

    for row in rows:
        if not _is_header_row(row):
            continue

        for orig_idx, w in row:
            text_lower = w["text"].lower()
            x = w["bbox"][0]

            if text_lower in HEADER_KEYWORDS["date"]:
                columns["date_x"] = x
            elif text_lower in HEADER_KEYWORDS["description"]:
                columns["desc_x"] = x
            elif text_lower in HEADER_KEYWORDS["amount"]:
                columns["amount_x"] = x
            elif text_lower in HEADER_KEYWORDS["balance"]:
                columns["balance_x"] = x

        break  # Use first header row found

    return columns


def _find_date_in_row(row: list) -> list:
    """Find date token indices in a row. Returns list of (position_in_row, orig_idx)."""
    dates = []
    for pos, (orig_idx, w) in enumerate(row):
        if _is_date_token(w["text"]):
            dates.append((pos, orig_idx))
            break  # Take first date only
        # Month name followed by day number
        if _is_month_name(w["text"]) and pos + 1 < len(row):
            next_text = row[pos + 1][1]["text"]
            if re.match(r"^\d{1,2},?$", next_text):
                dates.append((pos, orig_idx))
                break
    return dates


def _find_amounts_in_row(row: list) -> list:
    """Find amount token positions in a row. Returns list of (position_in_row, orig_idx, x_pos)."""
    amounts = []
    for pos, (orig_idx, w) in enumerate(row):
        if _is_amount_token(w["text"]):
            amounts.append((pos, orig_idx, w["bbox"][0]))
    return amounts


def tag_page(words: list) -> list:
    """Main tagging function. Returns list of BIO tags, one per word."""
    tags = ["O"] * len(words)

    if not words:
        return tags

    rows = _group_words_by_row(words)
    columns = _detect_columns(rows)

    # Track which rows are transaction rows (for multi-line desc detection)
    last_txn_row_idx = -1
    last_txn_desc_x = columns["desc_x"]

    for row_idx, row in enumerate(rows):
        # Skip section markers and headers
        if _is_section_marker(row) or _is_header_row(row):
            continue

        # Find date and amounts in this row
        dates = _find_date_in_row(row)
        amounts = _find_amounts_in_row(row)

        if dates and amounts:
            # This is a transaction row
            date_pos, date_orig = dates[0]

            # Tag date token(s)
            tags[date_orig] = "B-DATE"
            # Check for multi-token date (month name + day, or date parts)
            if _is_month_name(words[date_orig]["text"]):
                # Month name date — tag the day number as I-DATE
                if date_pos + 1 < len(row):
                    next_orig = row[date_pos + 1][0]
                    next_text = words[next_orig]["text"]
                    if re.match(r"^\d{1,2},?$", next_text):
                        tags[next_orig] = "I-DATE"
                        date_pos += 1  # Advance past date
                        # Check for year token
                        if date_pos + 1 < len(row):
                            year_orig = row[date_pos + 1][0]
                            year_text = words[year_orig]["text"]
                            if re.match(r"^\d{2,4}$", year_text):
                                tags[year_orig] = "I-DATE"
                                date_pos += 1

            # Sort amounts by X position (left to right)
            amounts.sort(key=lambda a: a[2])

            # Determine which amount is AMOUNT vs BALANCE
            if len(amounts) >= 2:
                # Rightmost is balance, others are amounts
                for i, (a_pos, a_orig, a_x) in enumerate(amounts):
                    if i == len(amounts) - 1:
                        tags[a_orig] = "B-BALANCE"
                    else:
                        tags[a_orig] = "B-AMOUNT"
            else:
                # Single amount
                tags[amounts[0][1]] = "B-AMOUNT"

            # Tag description — tokens between last date token and first amount
            first_amount_pos = amounts[0][0]
            desc_started = False

            for p in range(date_pos + 1, first_amount_pos):
                word_orig = row[p][0]
                if tags[word_orig] == "O":
                    if not desc_started:
                        tags[word_orig] = "B-DESC"
                        desc_started = True
                        last_txn_desc_x = words[word_orig]["bbox"][0]
                    else:
                        tags[word_orig] = "I-DESC"

            last_txn_row_idx = row_idx

        elif last_txn_row_idx == row_idx - 1 and not dates and not amounts:
            # Possible continuation row for multi-line description
            # Check if first word starts near the description column
            if row:
                first_x = row[0][1]["bbox"][0]
                # Must be roughly at description X position (within tolerance)
                if abs(first_x - last_txn_desc_x) < 80 and first_x > columns["date_x"] + 30:
                    for pos, (orig_idx, w) in enumerate(row):
                        # Stop if we hit an amount
                        if _is_amount_token(w["text"]):
                            break
                        tags[orig_idx] = "I-DESC"
                    last_txn_row_idx = row_idx  # Allow chaining

    return tags


def process_page(pdf_stem: str, page_idx: int, force: bool = False) -> dict:
    """Process a single page and write BIO output."""
    words_path = PAGES_DIR / pdf_stem / f"page_{page_idx}_words.json"
    output_path = TRAINING_DIR / pdf_stem / f"page_{page_idx}_bio.json"

    if not words_path.exists():
        return {"status": "missing"}

    if output_path.exists() and not force:
        return {"status": "skipped"}

    with open(words_path) as f:
        word_data = json.load(f)

    words = word_data.get("words", [])
    if not words:
        return {"status": "empty"}

    tags = tag_page(words)

    # Build output
    tag_counts = dict(Counter(tags))

    # Compute real quality score: a valid transaction needs B-DATE + B-AMOUNT + B-DESC.
    # Pages with no transactions or incomplete ones get lower scores.
    n_dates = tag_counts.get("B-DATE", 0)
    n_amounts = tag_counts.get("B-AMOUNT", 0)
    n_descs = tag_counts.get("B-DESC", 0)
    n_complete = min(n_dates, n_amounts, n_descs)  # fully-formed transactions

    if n_dates == 0:
        # No transactions found — could be a summary/cover page (valid but not useful)
        alignment_rate = 0.0
    elif n_dates == n_complete:
        # Every detected transaction has all 3 fields
        alignment_rate = 75.0  # still lower than Claude labels — rule-based
    else:
        # Some transactions missing fields
        alignment_rate = round(n_complete / n_dates * 75.0, 1)

    training_data = {
        "pdf_stem": pdf_stem,
        "page_index": page_idx,
        "words": [w["text"] for w in words],
        "bboxes": [w["bbox"] for w in words],
        "tags": tags,
        "word_count": len(words),
        "tag_counts": tag_counts,
        "alignment_rate": alignment_rate,
        "label_source": "rules",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    return {
        "status": "ok",
        "word_count": len(words),
        "tag_counts": tag_counts,
        "txn_count": tag_counts.get("B-DATE", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Rule-based BIO tagger for bank statements")
    parser.add_argument("--pdf", help="Label specific PDF stem")
    parser.add_argument("--force", action="store_true", help="Overwrite existing BIO files")
    parser.add_argument("--dry-run", action="store_true", help="Count pages without writing")
    parser.add_argument("--stats", action="store_true", help="Show tag distribution stats")
    args = parser.parse_args()

    if not PAGES_DIR.exists():
        print("No pages directory. Run pdf_to_pages.py first.")
        sys.exit(1)

    # Collect all page files
    if args.pdf:
        pdf_stems = [args.pdf]
    else:
        pdf_stems = sorted([d.name for d in PAGES_DIR.iterdir() if d.is_dir()])

    # Gather all (pdf_stem, page_idx) pairs
    pages = []
    for stem in pdf_stems:
        page_dir = PAGES_DIR / stem
        for wf in sorted(page_dir.glob("page_*_words.json")):
            page_idx = int(wf.stem.replace("page_", "").replace("_words", ""))
            pages.append((stem, page_idx))

    if args.dry_run:
        existing = sum(
            1 for stem, idx in pages
            if (TRAINING_DIR / stem / f"page_{idx}_bio.json").exists()
        )
        print(f"Total pages: {len(pages)}")
        print(f"Already labeled: {existing}")
        print(f"To label: {len(pages) - existing} (use --force to overwrite)")
        return

    print(f"Processing {len(pages)} pages from {len(pdf_stems)} PDFs...")
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    total_tagged = 0
    total_skipped = 0
    total_txns = 0
    all_tag_counts = Counter()

    for i, (stem, page_idx) in enumerate(pages):
        result = process_page(stem, page_idx, force=args.force)

        if result["status"] == "ok":
            total_tagged += 1
            total_txns += result.get("txn_count", 0)
            all_tag_counts.update(result.get("tag_counts", {}))
        elif result["status"] == "skipped":
            total_skipped += 1

        if (i + 1) % 200 == 0 or i == len(pages) - 1:
            print(f"  [{i+1}/{len(pages)}] Tagged: {total_tagged} | Skipped: {total_skipped} | Txns: {total_txns}")

    print(f"\nDone. Tagged {total_tagged} pages, {total_txns} transactions detected.")
    if total_skipped:
        print(f"Skipped {total_skipped} pages with existing BIO data (use --force to overwrite).")

    if args.stats or total_tagged > 0:
        print(f"\nTag distribution:")
        for tag, count in sorted(all_tag_counts.items(), key=lambda x: -x[1]):
            print(f"  {tag:<12} {count:>8}")


if __name__ == "__main__":
    main()
