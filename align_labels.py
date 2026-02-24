#!/usr/bin/env python3
"""
Align Claude Vision transaction labels to pdfplumber word tokens using BIO tagging.

For each page:
  1. Load Claude's transaction JSON (from label_with_claude.py)
  2. Load pdfplumber word tokens (from pdf_to_pages.py)
  3. Fuzzy-match date/amount tokens to find transaction anchors
  4. Tag description tokens between date and amount on same visual row
  5. Output BIO-tagged training data

BIO tags: O, B-DATE, I-DATE, B-DESC, I-DESC, B-AMOUNT, I-AMOUNT, B-BALANCE, I-BALANCE

Usage:
    python align_labels.py                    # align all labeled pages
    python align_labels.py --pdf app_123      # align specific PDF
    python align_labels.py --stats            # show alignment statistics
"""

import argparse
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

PROJECT_DIR = Path(__file__).resolve().parent
PAGES_DIR = PROJECT_DIR / "data" / "pages"
LABELS_DIR = PROJECT_DIR / "data" / "labels"
TRAINING_DIR = PROJECT_DIR / "data" / "training"

# Visual row grouping: words within this bbox-Y threshold are on the same row
ROW_Y_TOLERANCE = 8  # in normalized 0-1000 units (tighter to avoid merging rows)

# Fuzzy match thresholds
DATE_MATCH_THRESHOLD = 70  # lowered to catch more partial date matches
AMOUNT_MATCH_THRESHOLD = 80  # lowered slightly for format variations
DESC_MATCH_THRESHOLD = 60


def _normalize_amount(amount) -> str:
    """Normalize amount to comparable string."""
    try:
        val = abs(float(amount))
        return f"{val:,.2f}"
    except (ValueError, TypeError):
        return str(amount)


def _amount_variations(amount) -> list:
    """Generate variations of an amount string for matching."""
    try:
        val = abs(float(amount))
    except (ValueError, TypeError):
        return [str(amount)]

    variations = [
        f"{val:,.2f}",       # 1,234.56
        f"{val:.2f}",        # 1234.56
        f"${val:,.2f}",      # $1,234.56
        f"${val:.2f}",       # $1234.56
        f"{val:,.0f}",       # 1,235
    ]
    # Also handle negative formatting
    if float(amount) < 0:
        variations.extend([
            f"-{val:,.2f}",
            f"({val:,.2f})",
            f"-${val:,.2f}",
        ])
    return variations


def _date_variations(date_str: str) -> list:
    """Generate variations of a date string for matching."""
    if not date_str:
        return []
    variations = [date_str]

    # Strip year for short dates: "01/15/2024" → "01/15"
    parts = re.split(r"[/\-\s]+", date_str.strip())
    if len(parts) == 3:
        variations.append(f"{parts[0]}/{parts[1]}")
        variations.append(f"{parts[0]}-{parts[1]}")
        variations.append(f"{parts[0]}/{parts[1]}/{parts[2]}")

    # Month name variations
    month_map = {
        "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
        "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
        "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
    }
    # Reverse map: "Jan" → "01"
    name_to_num = {v: k for k, v in month_map.items()}
    name_to_num.update({v + ".": k for k, v in month_map.items()})

    if len(parts) >= 2 and parts[0] in month_map:
        variations.append(f"{month_map[parts[0]]} {parts[1]}")
        variations.append(f"{month_map[parts[0]]} {int(parts[1])}")

    # Handle "Jun 13", "Jun. 13", "June 13" — add just the month as a partial match
    if len(parts) >= 2 and (parts[0].capitalize() in name_to_num or
                             parts[0].capitalize().rstrip(".") in [m for m in name_to_num]):
        # Add the month name as standalone for multi-token matching
        variations.append(parts[0])
        # Add numeric form: "06/13"
        month_name = parts[0].capitalize().rstrip(".")
        if month_name in name_to_num:
            num = name_to_num[month_name]
            day = parts[1].zfill(2)
            variations.append(f"{num}/{day}")
            variations.append(f"{num}-{day}")

    # Handle full month names: "June" → "Jun"
    full_months = {
        "January": "Jan", "February": "Feb", "March": "Mar", "April": "Apr",
        "May": "May", "June": "Jun", "July": "Jul", "August": "Aug",
        "September": "Sep", "October": "Oct", "November": "Nov", "December": "Dec",
    }
    for full, abbr in full_months.items():
        if full.lower() in date_str.lower():
            for v in list(variations):
                variations.append(v.replace(full, abbr).replace(full.lower(), abbr.lower()))

    return list(set(variations))


def _group_words_by_row(words: list) -> list:
    """Group words into visual rows based on Y coordinate."""
    if not words:
        return []

    # Sort by top coordinate
    sorted_words = sorted(enumerate(words), key=lambda x: (x[1]["top"], x[1]["bbox"][0]))

    rows = []
    current_row = [sorted_words[0]]
    current_y = sorted_words[0][1]["top"]

    for idx_word in sorted_words[1:]:
        word_y = idx_word[1]["top"]
        if abs(word_y - current_y) <= ROW_Y_TOLERANCE:
            current_row.append(idx_word)
        else:
            # Sort row by x position
            current_row.sort(key=lambda x: x[1]["bbox"][0])
            rows.append(current_row)
            current_row = [idx_word]
            current_y = word_y

    if current_row:
        current_row.sort(key=lambda x: x[1]["bbox"][0])
        rows.append(current_row)

    return rows


def _find_token_match(words: list, target: str, variations: list, start_idx: int = 0) -> int:
    """Find the best matching word token index for a target string."""
    best_idx = -1
    best_score = 0

    for var in variations:
        for i in range(start_idx, len(words)):
            text = words[i]["text"]
            score = fuzz.ratio(text.lower(), var.lower())
            if score > best_score:
                best_score = score
                best_idx = i

            # Also try combining adjacent tokens
            if i + 1 < len(words):
                combined = f"{text} {words[i+1]['text']}"
                score = fuzz.ratio(combined.lower(), var.lower())
                if score > best_score:
                    best_score = score
                    best_idx = i

    return best_idx if best_score >= DATE_MATCH_THRESHOLD else -1


def align_page(words: list, transactions: list) -> list:
    """
    Align transactions to word tokens, producing BIO tags.

    Returns list of tags (same length as words), one tag per word.
    """
    tags = ["O"] * len(words)

    if not transactions or not words:
        return tags

    rows = _group_words_by_row(words)

    # Build row lookup: word_index → row_index
    word_to_row = {}
    for row_idx, row in enumerate(rows):
        for orig_idx, _ in row:
            word_to_row[orig_idx] = row_idx

    used_rows = set()

    for txn in transactions:
        date_str = txn.get("date", "")
        amount = txn.get("amount")
        description = txn.get("description", "")
        balance = txn.get("balance_after")

        # Step 1: Find date token (supports multi-token dates like "Jun" "13")
        date_vars = _date_variations(date_str)
        date_idx = -1
        best_date_score = 0

        for var in date_vars:
            for i, w in enumerate(words):
                # Single token match
                score = fuzz.ratio(w["text"].lower(), var.lower())
                if score > best_date_score and score >= DATE_MATCH_THRESHOLD:
                    row_idx = word_to_row.get(i, -1)
                    if row_idx not in used_rows:
                        best_date_score = score
                        date_idx = i

                # Multi-token match: combine current + next token
                if i + 1 < len(words) and word_to_row.get(i) == word_to_row.get(i + 1):
                    combined = f"{w['text']} {words[i+1]['text']}"
                    score = fuzz.ratio(combined.lower(), var.lower())
                    if score > best_date_score and score >= DATE_MATCH_THRESHOLD:
                        row_idx = word_to_row.get(i, -1)
                        if row_idx not in used_rows:
                            best_date_score = score
                            date_idx = i

        if date_idx < 0:
            continue

        date_row = word_to_row.get(date_idx, -1)
        if date_row < 0:
            continue

        # Step 2: Find amount token on same row or nearby rows
        amount_vars = _amount_variations(amount) if amount is not None else []
        amount_idx = -1
        best_amount_score = 0

        row_word_indices = set()
        for check_row in range(max(0, date_row), min(len(rows), date_row + 3)):
            for orig_idx, _ in rows[check_row]:
                row_word_indices.add(orig_idx)

        for var in amount_vars:
            for i in row_word_indices:
                # Clean OCR token: remove $, commas, trailing/leading hyphens, parens
                cleaned_token = (words[i]["text"]
                                 .replace(",", "").replace("$", "")
                                 .strip("-").strip("(").strip(")").lower())
                cleaned_var = (var.replace(",", "").replace("$", "")
                               .strip("-").strip("(").strip(")").lower())
                score = fuzz.ratio(cleaned_token, cleaned_var)
                if score > best_amount_score and score >= AMOUNT_MATCH_THRESHOLD:
                    best_amount_score = score
                    amount_idx = i

        if amount_idx < 0:
            continue

        # Mark this row as used
        used_rows.add(date_row)

        # Step 3: Tag date token(s)
        tags[date_idx] = "B-DATE"
        # Check if date spans multiple tokens (e.g., "Jan" "15")
        if date_idx + 1 < len(words) and word_to_row.get(date_idx + 1) == date_row:
            next_text = words[date_idx + 1]["text"]
            if re.match(r"^\d{1,2}$", next_text) or next_text in (",", "/", "-"):
                tags[date_idx + 1] = "I-DATE"
                # One more for year?
                if date_idx + 2 < len(words) and word_to_row.get(date_idx + 2) == date_row:
                    t = words[date_idx + 2]["text"]
                    if re.match(r"^\d{2,4}$", t) or t in (",", "/", "-"):
                        tags[date_idx + 2] = "I-DATE"

        # Step 4: Tag amount token
        tags[amount_idx] = "B-AMOUNT"

        # Step 5: Tag description — tokens between date and amount on same row
        row_words = rows[date_row]
        row_indices = [idx for idx, _ in row_words]

        if date_idx in row_indices and amount_idx in row_indices:
            date_pos = row_indices.index(date_idx)
            amount_pos = row_indices.index(amount_idx)

            # Find the last date token position
            last_date_pos = date_pos
            for p in range(date_pos + 1, amount_pos):
                if tags[row_indices[p]] in ("I-DATE",):
                    last_date_pos = p
                else:
                    break

            # Description = tokens between last date token and amount
            desc_started = False
            for p in range(last_date_pos + 1, amount_pos):
                word_idx = row_indices[p]
                if tags[word_idx] == "O":
                    if not desc_started:
                        tags[word_idx] = "B-DESC"
                        desc_started = True
                    else:
                        tags[word_idx] = "I-DESC"

        # Step 6: Tag balance (if present, usually after amount on same row)
        if balance is not None:
            balance_vars = _amount_variations(balance)
            best_bal_score = 0
            balance_idx = -1

            for var in balance_vars:
                for i in row_word_indices:
                    if i == amount_idx or tags[i] != "O":
                        continue
                    cleaned_token = (words[i]["text"]
                                     .replace(",", "").replace("$", "")
                                     .strip("-").strip("(").strip(")").lower())
                    cleaned_var = (var.replace(",", "").replace("$", "")
                                   .strip("-").strip("(").strip(")").lower())
                    score = fuzz.ratio(cleaned_token, cleaned_var)
                    if score > best_bal_score and score >= AMOUNT_MATCH_THRESHOLD:
                        best_bal_score = score
                        balance_idx = i

            if balance_idx >= 0:
                tags[balance_idx] = "B-BALANCE"

    return tags


def process_page(pdf_stem: str, page_idx: int) -> dict:
    """Align labels for a single page. Returns alignment stats."""
    words_path = PAGES_DIR / pdf_stem / f"page_{page_idx}_words.json"
    labels_path = LABELS_DIR / pdf_stem / f"page_{page_idx}_labels.json"

    if not words_path.exists() or not labels_path.exists():
        return {"status": "missing", "aligned": 0, "total_txns": 0}

    with open(words_path) as f:
        word_data = json.load(f)
    with open(labels_path) as f:
        label_data = json.load(f)

    words = word_data.get("words", [])
    transactions = label_data.get("transactions", [])

    if not transactions:
        # No transactions on this page — all O tags
        tags = ["O"] * len(words)
        aligned_count = 0
    else:
        tags = align_page(words, transactions)
        # Count how many transactions were successfully aligned (have at least B-DATE + B-AMOUNT)
        date_count = sum(1 for t in tags if t == "B-DATE")
        amount_count = sum(1 for t in tags if t == "B-AMOUNT")
        aligned_count = min(date_count, amount_count)

    # Save training data
    training_dir = TRAINING_DIR / pdf_stem
    training_dir.mkdir(parents=True, exist_ok=True)

    training_data = {
        "pdf_stem": pdf_stem,
        "page_index": page_idx,
        "words": [w["text"] for w in words],
        "bboxes": [w["bbox"] for w in words],
        "tags": tags,
        "word_count": len(words),
        "tag_counts": {},
        "alignment_rate": round(aligned_count / len(transactions) * 100, 1) if transactions else 100.0,
        "label_source": label_data.get("label_source", "claude"),
    }

    # Tag distribution
    from collections import Counter
    training_data["tag_counts"] = dict(Counter(tags))

    output_path = training_dir / f"page_{page_idx}_bio.json"
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    return {
        "status": "ok",
        "aligned": aligned_count,
        "total_txns": len(transactions),
        "word_count": len(words),
        "tag_counts": training_data["tag_counts"],
        "alignment_rate": training_data["alignment_rate"],
    }


def main():
    parser = argparse.ArgumentParser(description="Align Claude labels to BIO token tags")
    parser.add_argument("--pdf", help="Align specific PDF stem")
    parser.add_argument("--stats", action="store_true", help="Show alignment statistics only")
    args = parser.parse_args()

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdf_stems = [args.pdf]
    else:
        if not LABELS_DIR.exists():
            print(f"No labels directory. Run label_with_claude.py first.")
            sys.exit(1)
        pdf_stems = sorted([d.name for d in LABELS_DIR.iterdir() if d.is_dir()])

    if not pdf_stems:
        print("No labeled PDFs found.")
        sys.exit(1)

    print(f"Aligning {len(pdf_stems)} PDFs...")

    total_pages = 0
    total_txns = 0
    total_aligned = 0
    alignment_rates = []

    for i, stem in enumerate(pdf_stems):
        label_dir = LABELS_DIR / stem
        label_files = sorted(label_dir.glob("page_*_labels.json"))

        pdf_aligned = 0
        pdf_txns = 0

        for lf in label_files:
            page_idx = int(lf.stem.replace("page_", "").replace("_labels", ""))
            result = process_page(stem, page_idx)

            if result["status"] == "ok":
                total_pages += 1
                total_txns += result["total_txns"]
                total_aligned += result["aligned"]
                pdf_aligned += result["aligned"]
                pdf_txns += result["total_txns"]
                alignment_rates.append(result["alignment_rate"])

        if (i + 1) % 25 == 0 or i == len(pdf_stems) - 1:
            rate = round(total_aligned / total_txns * 100, 1) if total_txns > 0 else 0
            print(f"  [{i+1}/{len(pdf_stems)}] Pages: {total_pages}, "
                  f"Aligned: {total_aligned}/{total_txns} ({rate}%)")

    # Final summary
    overall_rate = round(total_aligned / total_txns * 100, 1) if total_txns > 0 else 0
    avg_page_rate = round(sum(alignment_rates) / len(alignment_rates), 1) if alignment_rates else 0

    print(f"\nAlignment Summary:")
    print(f"  PDFs processed:     {len(pdf_stems)}")
    print(f"  Pages processed:    {total_pages}")
    print(f"  Total transactions: {total_txns}")
    print(f"  Aligned:            {total_aligned} ({overall_rate}%)")
    print(f"  Avg page alignment: {avg_page_rate}%")

    if overall_rate < 90:
        print(f"\n  WARNING: Alignment rate ({overall_rate}%) is below 90% target.")
        print(f"  Consider reviewing low-alignment pages with review_labels.py")


if __name__ == "__main__":
    main()
