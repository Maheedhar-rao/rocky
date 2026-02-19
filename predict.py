#!/usr/bin/env python3
"""
LayoutLMv3 inference for bank statement parsing.

Singleton model with thread-safe loading. Takes PDF bytes → structured transactions.

Usage as module:
    from predict import get_parser, reload_parser
    parser = get_parser()
    result = parser.parse_pdf(pdf_bytes)

Usage as CLI:
    python predict.py statement.pdf
    python predict.py statement.pdf --json
"""

import argparse
import io
import json
import re
import statistics
import sys
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pdfplumber
import torch
from pdf2image import convert_from_bytes
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "statement_parser"
FEEDBACK_DIR = PROJECT_DIR / "data" / "feedback"

MAX_SEQ_LEN = 512
MIN_WORDS_THRESHOLD = 20

# BIO tags — loaded from meta.json if available
BIO_TAGS = ["O", "B-DATE", "I-DATE", "B-DESC", "I-DESC", "B-AMOUNT", "I-AMOUNT", "B-BALANCE", "I-BALANCE"]
TAG2ID = {tag: i for i, tag in enumerate(BIO_TAGS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}


@dataclass
class Transaction:
    date: str
    description: str
    amount: float
    type: str  # "credit" or "debit"
    balance_after: Optional[float] = None
    page: int = 0
    confidence: float = 0.0


@dataclass
class ParseResult:
    transactions: list
    total_deposits: float = 0.0
    total_withdrawals: float = 0.0
    average_daily_balance: Optional[float] = None
    nsf_count: int = 0
    page_count: int = 0
    word_count: int = 0
    model_version: str = ""
    low_confidence_pages: list = None


def _normalize_bbox(bbox: tuple, page_width: float, page_height: float) -> list:
    x0, y0, x1, y1 = bbox
    return [
        int(max(0, min(1000, x0 / page_width * 1000))),
        int(max(0, min(1000, y0 / page_height * 1000))),
        int(max(0, min(1000, x1 / page_width * 1000))),
        int(max(0, min(1000, y1 / page_height * 1000))),
    ]


def _extract_words_pdfplumber(page) -> list:
    words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    pw, ph = page.width, page.height
    result = []
    for w in words:
        text = w["text"].strip()
        if not text:
            continue
        bbox = _normalize_bbox((w["x0"], w["top"], w["x1"], w["bottom"]), pw, ph)
        result.append({"text": text, "bbox": bbox})
    return result


def _extract_words_ocr(image: Image.Image) -> list:
    import pytesseract
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    iw, ih = image.size
    result = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        if not text or int(data["conf"][i]) < 30:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        bbox = _normalize_bbox((x, y, x + w, y + h), iw, ih)
        result.append({"text": text, "bbox": bbox})
    return result


def _parse_amount(text: str) -> float:
    if not text:
        return 0.0
    cleaned = re.sub(r"[,$()\\s]", "", text)
    cleaned = cleaned.replace("−", "-").replace("–", "-")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _bio_to_transactions(words: list, tags: list, page_idx: int, confidences: list = None) -> list:
    """Convert BIO-tagged tokens to Transaction objects."""
    transactions = []
    current = {}
    current_conf = []

    for i, (word, tag) in enumerate(zip(words, tags)):
        conf = confidences[i] if confidences and i < len(confidences) else 0.0

        if tag.startswith("B-"):
            field = tag[2:].lower()
            if field == "date" and current and "date" in current:
                # Flush previous transaction
                txn = _build_transaction(current, current_conf, page_idx)
                if txn:
                    transactions.append(txn)
                current = {}
                current_conf = []

            current[field] = word
            current_conf.append(conf)

        elif tag.startswith("I-"):
            field = tag[2:].lower()
            if field in current:
                # Amounts/balances never have spaces (e.g., "1,234" + ".56")
                # Dates: no space for separators/digits after separators,
                #        but space for "Feb" + "1" style dates
                # Descriptions always get spaces
                if field in ("amount", "balance"):
                    sep = ""
                elif field == "date":
                    prev = current[field]
                    sep = "" if prev.endswith(("/", "-")) or word in ("/", "-", ",") else " "
                else:
                    sep = " "
                current[field] += f"{sep}{word}"
            elif current:
                # I-tag without B-tag: promote to B-tag behavior.
                # This handles chunk boundary misalignment where the model
                # correctly identifies continuation tokens but missed the B-tag.
                current[field] = word
            current_conf.append(conf)

    # Flush last transaction
    if current and "date" in current:
        txn = _build_transaction(current, current_conf, page_idx)
        if txn:
            transactions.append(txn)

    return transactions


def _build_transaction(fields: dict, confidences: list, page_idx: int) -> Optional[Transaction]:
    """Build a Transaction from extracted fields."""
    date = fields.get("date", "").strip()
    desc = fields.get("desc", "").strip()
    amount_str = fields.get("amount", "0")
    balance_str = fields.get("balance")

    amount = _parse_amount(amount_str)
    if amount == 0.0 and not amount_str.strip():
        return None

    balance = _parse_amount(balance_str) if balance_str else None

    # Determine credit/debit from explicit sign only
    # Full inference happens later in _infer_transaction_types()
    txn_type = "debit" if amount < 0 else "unknown"

    avg_conf = statistics.mean(confidences) if confidences else 0.0

    return Transaction(
        date=date,
        description=desc,
        amount=abs(amount),
        type=txn_type,
        balance_after=balance,
        page=page_idx,
        confidence=round(avg_conf, 4),
    )


CONFIDENCE_THRESHOLD = 0.75  # transactions below this get flagged

# Common bank identifiers for feedback grouping
_BANK_PATTERNS = {
    "chase": "Chase", "jpmorgan": "Chase",
    "wells fargo": "Wells Fargo", "wellsfargo": "Wells Fargo",
    "bank of america": "Bank of America", "bankofamerica": "Bank of America",
    "citibank": "Citibank", "citi ": "Citibank",
    "us bank": "US Bank", "u.s. bank": "US Bank",
    "pnc": "PNC", "td bank": "TD Bank",
    "capital one": "Capital One", "truist": "Truist",
    "regions": "Regions", "keybank": "KeyBank",
    "comerica": "Comerica", "citizens": "Citizens",
    "navy federal": "Navy Federal", "usaa": "USAA",
    "huntington": "Huntington", "m&t bank": "M&T Bank",
    "fifth third": "Fifth Third", "zions": "Zions",
    "bbva": "BBVA", "bmo": "BMO",
}


def _detect_bank_format(page_words: dict) -> str:
    """Detect bank format from first-page text."""
    if not page_words or 0 not in page_words:
        return "unknown"
    text = " ".join(w["text"].lower() for w in page_words[0][:100])
    for pattern, bank in _BANK_PATTERNS.items():
        if pattern in text:
            return bank
    return "unknown"


def _log_feedback(pdf_hash: str, transactions: list, low_conf_pages: list,
                  bank_format: str, page_count: int):
    """Log low-confidence predictions to data/feedback/ for review."""
    flagged_txns = [t for t in transactions if t.get("confidence", 1.0) < CONFIDENCE_THRESHOLD]
    if not flagged_txns and not low_conf_pages:
        return

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_path = FEEDBACK_DIR / f"{pdf_hash}.json"
    if feedback_path.exists():
        return  # already flagged

    feedback = {
        "pdf_hash": pdf_hash,
        "bank_format": bank_format,
        "page_count": page_count,
        "total_transactions": len(transactions),
        "flagged_transactions": len(flagged_txns),
        "low_confidence_pages": low_conf_pages,
        "avg_confidence": round(
            statistics.mean(t.get("confidence", 0) for t in transactions), 4
        ) if transactions else 0,
        "flagged_details": [
            {"page": t.get("page", 0), "description": t.get("description", "")[:50],
             "confidence": t.get("confidence", 0)}
            for t in flagged_txns[:20]  # cap at 20 for storage
        ],
        "timestamp": datetime.now().isoformat(),
    }

    with open(feedback_path, "w") as f:
        json.dump(feedback, f, indent=2)


# Keywords that indicate a debit section
_DEBIT_SECTION_KW = {
    "withdrawal", "withdrawals", "debit", "debits", "checks",
    "check", "atm", "electronic withdrawal", "other withdrawal",
    "purchases", "fees", "service charge",
}
# Keywords that indicate a credit section
_CREDIT_SECTION_KW = {
    "deposit", "deposits", "credit", "credits", "addition", "additions",
}


def _classify_credit_debit(transactions: list, page_words: dict = None) -> list:
    """Classify transactions as credit/debit using hybrid approach:
    1. Sign-based (already set in _build_transaction for negative amounts)
    2. Balance-change method (heuristic Pass 1)
    3. Section-heading method (heuristic Pass 2)
    4. FFN classifier for remaining unknowns (replaces keyword + default)
    """
    if not transactions:
        return transactions

    # Pass 1 & 2: run balance-change and section-heading heuristics
    # (these only touch "unknown" types, preserving sign-based detection)
    _infer_transaction_types_passes_1_2(transactions, page_words)

    # Pass 3: FFN for remaining unknowns
    unknowns = [t for t in transactions if t.type == "unknown"]
    if unknowns:
        try:
            from credit_debit_model import get_classifier
            clf = get_classifier()
            txn_dicts = [{"amount": t.amount, "description": t.description, "page": t.page}
                         for t in unknowns]
            results = clf.predict(txn_dicts)
            for t, (pred_type, confidence) in zip(unknowns, results):
                t.type = pred_type
        except (FileNotFoundError, ImportError, Exception):
            # FFN not available — keyword fallback
            _infer_transaction_types_pass_3_4(unknowns)

    return transactions


def _infer_transaction_types_passes_1_2(transactions: list, page_words: dict = None) -> list:
    """Passes 1-2: Balance-change and section-heading inference (high accuracy)."""
    if not transactions:
        return transactions

    # --- Pass 1: Balance-change inference ---
    for i in range(len(transactions)):
        t = transactions[i]
        if t.type != "unknown":
            continue

        if t.balance_after is not None and i > 0:
            prev_bal = None
            for j in range(i - 1, -1, -1):
                if transactions[j].balance_after is not None:
                    prev_bal = transactions[j].balance_after
                    break

            if prev_bal is not None:
                diff = t.balance_after - prev_bal
                if abs(diff + t.amount) < 1.0:  # balance dropped
                    t.type = "debit"
                elif abs(diff - t.amount) < 1.0:  # balance rose
                    t.type = "credit"

    # --- Pass 2: Section heading inference from page words ---
    if page_words:
        page_sections = _detect_page_sections(page_words)
        for t in transactions:
            if t.type != "unknown":
                continue
            sections_on_page = page_sections.get(t.page, [])
            words = page_words.get(t.page, [])
            section = _detect_section_for_txn(t, sections_on_page, words)
            if section:
                t.type = section

    return transactions


def _infer_transaction_types_pass_3_4(transactions: list) -> list:
    """Passes 3-4: Keyword fallback + default (used when FFN is unavailable)."""
    credit_kw = ("deposit", "direct dep", "payroll", "salary", "wages",
                 "credit", "transfer in", "wire in", "refund",
                 "interest paid", "ach credit")
    debit_kw = ("fee", "service charge", "bill payment", "loan payment",
                "ach debit", "withdrawal", "purchase",
                "check paid", "check #", "wire out", "transfer out")

    for t in transactions:
        if t.type != "unknown":
            continue
        desc_lower = t.description.lower()
        if any(kw in desc_lower for kw in credit_kw):
            t.type = "credit"
        elif any(kw in desc_lower for kw in debit_kw):
            t.type = "debit"

    for t in transactions:
        if t.type == "unknown":
            t.type = "debit"  # default to debit (more common, safer)

    return transactions


def _infer_transaction_types(transactions: list, page_words: dict = None) -> list:
    """Full heuristic fallback (all 4 passes). Used when _classify_credit_debit isn't called."""
    _infer_transaction_types_passes_1_2(transactions, page_words)
    _infer_transaction_types_pass_3_4(transactions)
    return transactions


def _detect_page_sections(page_words: dict) -> dict:
    """Detect credit/debit sections for each page from section headings.

    Returns: {page_idx: [(word_position, section_type), ...]}
    Handles both single-word headers ("Deposits") and multi-word headers
    ("Withdrawals and other debits", "Other Credits (+)").
    """
    DEBIT_PHRASES = [
        "withdrawals and other debits", "withdrawals and other debits - continued",
        "other withdrawals", "electronic withdrawals", "checks paid",
        "checks(-)", "other debits", "atm & debit card",
    ]
    CREDIT_PHRASES = [
        "deposits and other credits", "deposits and other credits - continued",
        "other credits", "electronic deposits",
        "credits(+)", "other credits (+)",
    ]

    page_sections = {}
    for page_idx, words in page_words.items():
        sections = []
        texts = [w["text"].lower() for w in words]
        full_text = " ".join(texts)

        # Multi-word phrase detection
        for phrase in DEBIT_PHRASES:
            if phrase in full_text:
                # Find approximate word position
                pos = full_text.index(phrase)
                word_pos = full_text[:pos].count(" ")
                sections.append((word_pos, "debit"))

        for phrase in CREDIT_PHRASES:
            if phrase in full_text:
                pos = full_text.index(phrase)
                word_pos = full_text[:pos].count(" ")
                sections.append((word_pos, "credit"))

        # Single-word detection
        for i, text_lower in enumerate(texts):
            if text_lower in _DEBIT_SECTION_KW:
                # Avoid duplicates near multi-word matches
                if not any(abs(sp - i) < 5 for sp, _ in sections):
                    sections.append((i, "debit"))
            elif text_lower in _CREDIT_SECTION_KW:
                if not any(abs(sp - i) < 5 for sp, _ in sections):
                    sections.append((i, "credit"))

        sections.sort(key=lambda x: x[0])
        page_sections[page_idx] = sections

    return page_sections


def _detect_section_for_txn(txn, sections_on_page: list, words: list) -> Optional[str]:
    """Determine if a transaction is in a credit or debit section.

    Uses the transaction's date string to find its approximate position in the
    word list, then looks at which section heading appears before it.
    """
    if not sections_on_page or not words:
        return None

    # If page has only one section type, all txns on that page inherit it
    section_types = set(s[1] for s in sections_on_page)
    if len(section_types) == 1:
        return section_types.pop()

    # Find the transaction's position by matching its date token
    txn_pos = None
    date_str = txn.date.lower().replace("/", "/")

    # Try matching date first, then description first word
    for i, w in enumerate(words):
        if w["text"].lower() == date_str:
            txn_pos = i
            break

    if txn_pos is None and txn.description:
        first_word = txn.description.lower().split()[0] if txn.description.strip() else ""
        if first_word:
            for i, w in enumerate(words):
                if w["text"].lower() == first_word:
                    txn_pos = i
                    break

    if txn_pos is None:
        return None

    # Find closest section heading before this transaction
    best_section = None
    for sec_pos, sec_type in sections_on_page:
        if sec_pos < txn_pos:
            best_section = sec_type

    return best_section


def _filter_non_txn_sections(words: list) -> list:
    """
    Remove words from non-transaction sections at both the TOP and BOTTOM of a page.

    TOP: Bank statements often start with summary sections (CHECKING SUMMARY, fee info)
    before the actual TRANSACTION DETAIL section. These waste token budget.

    BOTTOM: Statements commonly end with "Daily Balance Summary" sections whose rows
    look like transactions (date + amount) but aren't.

    Works for any bank — detects section boundaries by scanning for known header phrases.
    """
    if not words:
        return words

    texts = [w["text"].lower() for w in words]

    # --- Step 1: Find start of transaction section (remove pre-transaction preamble) ---
    TXN_START_HEADERS = [
        "transaction detail",
        "transaction activity",
        "account activity",
        "itemized transactions",
        "transaction history",
    ]

    start_idx = 0
    for header in TXN_START_HEADERS:
        header_tokens = header.split()
        hlen = len(header_tokens)
        for i in range(len(texts) - hlen + 1):
            window = " ".join(texts[i : i + hlen])
            if window == header:
                if i > start_idx:
                    start_idx = i
                break

    # Only cut pre-content if there's a significant preamble (> 20 words)
    if start_idx < 20:
        start_idx = 0

    # --- Step 2: Find end of transaction section (remove post-transaction content) ---
    NON_TXN_HEADERS = [
        "daily balance summary",
        "daily ledger balance",
        "daily ledger balances",
        "daily ending balance",
        "daily ending balances",
        "ending daily balance",
        "ending daily balances",
        "average ledger balance",
        "balance summary",
        "balance activity",
        "daily balance",
        "daily balances",
        "service charge summary",
        "total fees charged",
        "year to date totals",
        "account balance summary",
    ]

    cutoff_idx = len(words)  # default: keep all

    for header in NON_TXN_HEADERS:
        header_tokens = header.split()
        hlen = len(header_tokens)
        for i in range(start_idx, len(texts) - hlen + 1):
            window = " ".join(texts[i : i + hlen])
            if window == header:
                if i < cutoff_idx:
                    cutoff_idx = i
                break

    return words[start_idx:cutoff_idx]


def _extract_embedded_amounts(transactions: list) -> list:
    """
    Fix transactions where the model put the dollar amount inside the description.

    This happens when the model can't separate amount columns from description columns
    (common in unfamiliar layouts). If a transaction has amount=0.0 but the description
    contains a pattern like "-$17.75" or "$500.00", extract it.

    Works for any bank — purely pattern-based on dollar-sign notation.
    """
    for txn in transactions:
        if txn.amount != 0.0:
            continue
        if not txn.description:
            continue

        # Find all dollar amounts in the description: -$17.75, $500.00, −$6.73
        matches = re.findall(r'[-−–]?\$[\d,]+\.?\d*', txn.description)
        if not matches:
            continue

        # Take the LAST dollar amount (typically the amount, not a reference number)
        raw = matches[-1]
        cleaned = raw.replace('$', '').replace(',', '').replace('−', '-').replace('–', '-')
        try:
            parsed = float(cleaned)
        except ValueError:
            continue

        txn.amount = abs(parsed)
        if parsed < 0:
            txn.type = "debit"

        # Remove all embedded dollar amounts from description to clean it up
        for m in matches:
            txn.description = txn.description.replace(m, '')
        txn.description = re.sub(r'\s+', ' ', txn.description).strip()

    return transactions


def _filter_summary_rows(transactions: list) -> list:
    """Remove transactions that are actually summary/header rows.

    Common false positives:
    - Account summary amounts (Beginning/Ending Balance, Total Credits/Debits)
    - Section totals ("Total deposits and other credits $568,244.71")
    - Description contains summary keywords rather than a real transaction
    """
    if not transactions:
        return transactions

    SUMMARY_KEYWORDS = [
        "beginning balance", "ending balance", "opening balance", "closing balance",
        "total deposits", "total withdrawals", "total credits", "total debits",
        "total checks", "total fees", "total service",
        "credits (+)", "debits (-)", "credits(+)", "debits(-)",
        "account summary", "balance forward", "previous balance",
        "average ledger", "average collected", "average daily",
        "statement period", "days in cycle", "days in statement",
        "number of deposits", "number of withdrawals",
        "# of deposits", "# of withdrawals",
        "subtotal for card", "total for card",
    ]

    filtered = []
    for txn in transactions:
        desc_lower = txn.description.lower().strip()

        # Check if description matches summary keywords
        is_summary = False
        for kw in SUMMARY_KEYWORDS:
            if kw in desc_lower:
                is_summary = True
                break

        # Empty/very short descriptions with no meaningful text are likely
        # summary rows, daily balance entries, or parsing artifacts.
        # BUT: keep transactions with non-zero amounts — these may be real
        # transactions where the model failed to extract the description
        # (e.g., at chunk boundaries or page edges).
        if len(desc_lower) < 3 and txn.amount == 0.0:
            is_summary = True

        if not is_summary:
            filtered.append(txn)

    return filtered


class StatementParserLocal:
    """Thread-safe LayoutLMv3 singleton for bank statement parsing."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(MODEL_DIR)
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.version = "unknown"
        self._load()

    def _load(self):
        model_dir = Path(self.model_path)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"No model at {model_dir}. Run train.py first.")

        self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_dir))
        self.processor = LayoutLMv3Processor.from_pretrained(str(model_dir), apply_ocr=False)
        self.model.to(self.device)
        self.model.eval()

        meta_path = model_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.version = json.load(f).get("version", "unknown")

    def _predict_chunk(self, word_texts: list, word_bboxes: list, image: Image.Image) -> tuple:
        """Run inference on a single chunk of words. Returns (tags, confidences)."""
        encoding = self.processor(
            image, word_texts, boxes=word_bboxes,
            truncation=True, max_length=MAX_SEQ_LEN, padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            device_encoding = {k: v.to(self.device) for k, v in encoding.items()}
            outputs = self.model(**device_encoding)
            logits = outputs.logits.squeeze(0).cpu()
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            max_probs = probs.max(dim=-1).values

        # Map sub-tokens back to original words using word_ids
        tags = ["O"] * len(word_texts)
        confidences = [0.0] * len(word_texts)

        word_ids = encoding.word_ids(batch_index=0)
        seen = set()
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            if wid < len(word_texts):
                tag_id = predictions[idx].item()
                tags[wid] = ID2TAG.get(tag_id, "O")
                confidences[wid] = max_probs[idx].item()

        return tags, confidences

    def _predict_page(self, words: list, bboxes: list, image: Image.Image) -> tuple:
        """Run inference on a single page with automatic chunking for dense pages.

        LayoutLMv3 has a 512 sub-token limit. Long words (reference numbers, URLs)
        can expand to many sub-tokens, so 400 words might exceed the limit. This
        method detects when truncation occurs and splits into overlapping chunks.
        """
        if not words:
            return [], []

        all_texts = [w["text"] for w in words]
        all_bboxes = [w["bbox"] for w in words]

        # First pass: try processing all words at once
        chunk_texts = all_texts[:MAX_SEQ_LEN]
        chunk_bboxes = all_bboxes[:MAX_SEQ_LEN]

        tags, confidences = self._predict_chunk(chunk_texts, chunk_bboxes, image)

        # Check how many words actually fit in 512 tokens
        test_enc = self.processor(
            image, chunk_texts, boxes=chunk_bboxes,
            truncation=True, max_length=MAX_SEQ_LEN, padding="max_length",
            return_tensors="pt",
        )
        word_ids = test_enc.word_ids(batch_index=0)
        mapped_ids = [wid for wid in word_ids if wid is not None]
        max_mapped = max(mapped_ids) if mapped_ids else 0

        # If all words fit, return directly
        if max_mapped >= len(all_texts) - 1:
            return tags, confidences

        # Dense page: need chunked inference
        # Determine safe chunk size based on how many words fit
        words_per_chunk = max(max_mapped - 10, 50)  # leave margin
        overlap = 20  # overlap to avoid splitting transactions at boundaries

        all_tags = ["O"] * len(all_texts)
        all_confs = [0.0] * len(all_texts)

        # Copy first chunk results (only mapped words)
        for i in range(min(max_mapped + 1, len(all_tags))):
            all_tags[i] = tags[i]
            all_confs[i] = confidences[i]

        # Track the global frontier: last word index that was actually mapped
        global_frontier = max_mapped

        # Process remaining chunks
        start = max(global_frontier + 1 - overlap, 0)
        max_iterations = 20  # safety limit
        for _ in range(max_iterations):
            if start >= len(all_texts):
                break

            end = min(start + words_per_chunk, len(all_texts))
            chunk_t = all_texts[start:end]
            chunk_b = all_bboxes[start:end]

            if not chunk_t:
                break

            c_tags, c_confs = self._predict_chunk(chunk_t, chunk_b, image)

            # Find the last word actually mapped in this chunk
            # (unmapped words have conf=0.0 and tag="O")
            chunk_last_mapped = -1
            for i in range(len(c_confs) - 1, -1, -1):
                if c_confs[i] > 0:
                    chunk_last_mapped = i
                    break

            if chunk_last_mapped < 0:
                break  # chunk produced nothing useful

            # Merge: keep higher confidence prediction for all mapped words
            for i in range(chunk_last_mapped + 1):
                global_i = start + i
                if global_i >= len(all_tags):
                    break
                if c_confs[i] > all_confs[global_i]:
                    all_tags[global_i] = c_tags[i]
                    all_confs[global_i] = c_confs[i]

            # Update frontier
            new_frontier = start + chunk_last_mapped
            if new_frontier <= global_frontier:
                break  # no forward progress
            global_frontier = new_frontier

            if global_frontier >= len(all_texts) - 1:
                break  # all words covered

            start = max(global_frontier + 1 - overlap, start + 1)

        return all_tags, all_confs

    def parse_pdf(self, pdf_bytes: bytes) -> ParseResult:
        """Parse a full PDF and return structured transactions + metrics."""
        # Convert to images
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200)
        except Exception as e:
            return ParseResult(
                transactions=[],
                page_count=0,
                model_version=self.version,
                low_confidence_pages=[],
            )

        # Extract words per page
        all_transactions = []
        total_words = 0
        low_conf_pages = []
        page_words = {}  # page_idx -> word list (for section inference)

        try:
            pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
        except Exception:
            pdf = None

        for page_idx, image in enumerate(images):
            # Extract words
            words = []
            if pdf and page_idx < len(pdf.pages):
                pdfp_words = pdf.pages[page_idx].extract_words(
                    x_tolerance=3, y_tolerance=3, keep_blank_chars=False
                )
                pw = pdf.pages[page_idx].width
                ph = pdf.pages[page_idx].height
                for w in pdfp_words:
                    text = w["text"].strip()
                    if text:
                        bbox = _normalize_bbox((w["x0"], w["top"], w["x1"], w["bottom"]), pw, ph)
                        words.append({"text": text, "bbox": bbox})

            if len(words) < MIN_WORDS_THRESHOLD:
                ocr_words = _extract_words_ocr(image.convert("RGB"))
                if len(ocr_words) > len(words):
                    words = ocr_words

            total_words += len(words)
            page_words[page_idx] = words  # full words for section inference

            # Pre-filter: remove words from non-transaction sections
            # (e.g., "Daily Balance Summary" at bottom of page)
            txn_words = _filter_non_txn_sections(words)

            if not txn_words:
                continue

            # Run LayoutLMv3 inference on transaction-section words only
            tags, confidences = self._predict_page(txn_words, txn_words, image.convert("RGB"))

            # Check page confidence
            non_o_confs = [c for t, c in zip(tags, confidences) if t != "O"]
            avg_conf = statistics.mean(non_o_confs) if non_o_confs else 1.0
            if avg_conf < 0.70:
                low_conf_pages.append(page_idx)

            # Convert BIO tags to transactions
            word_texts = [w["text"] for w in txn_words[:len(tags)]]
            page_txns = _bio_to_transactions(word_texts, tags, page_idx, confidences)
            all_transactions.extend(page_txns)

        if pdf:
            pdf.close()

        # Post-processing: extract amounts embedded in descriptions ($0 amount fix)
        _extract_embedded_amounts(all_transactions)

        # Post-processing: drop transactions with no usable data
        all_transactions = [
            t for t in all_transactions
            if t.amount != 0.0 or t.balance_after is not None
        ]

        # Post-processing: remove likely summary/header rows
        all_transactions = _filter_summary_rows(all_transactions)

        # Infer credit/debit types: use FFN classifier if available, else heuristic
        _classify_credit_debit(all_transactions, page_words)

        # Build result dicts for feedback + return
        txn_dicts = [asdict(t) for t in all_transactions]

        # Feedback logging: flag low-confidence extractions
        import hashlib
        pdf_hash = hashlib.md5(pdf_bytes[:4096]).hexdigest()[:12]
        bank_format = _detect_bank_format(page_words)
        try:
            _log_feedback(pdf_hash, txn_dicts, low_conf_pages, bank_format, len(images))
        except Exception:
            pass  # feedback logging should never break parsing

        # Compute summary metrics
        total_deposits = sum(t.amount for t in all_transactions if t.type == "credit")
        total_withdrawals = sum(t.amount for t in all_transactions if t.type == "debit")

        # NSF detection (common patterns)
        nsf_count = sum(
            1 for t in all_transactions
            if any(kw in t.description.upper() for kw in ("NSF", "INSUFFICIENT", "OVERDRAFT", "OD FEE", "RETURNED ITEM"))
        )

        # Average daily balance from balance_after values
        balances = [t.balance_after for t in all_transactions if t.balance_after is not None]
        adb = statistics.mean(balances) if balances else None

        return ParseResult(
            transactions=txn_dicts,
            total_deposits=round(total_deposits, 2),
            total_withdrawals=round(total_withdrawals, 2),
            average_daily_balance=round(adb, 2) if adb else None,
            nsf_count=nsf_count,
            page_count=len(images),
            word_count=total_words,
            model_version=self.version,
            low_confidence_pages=low_conf_pages or [],
        )


# Thread-safe singleton
_parser = None
_lock = threading.Lock()


def get_parser() -> StatementParserLocal:
    global _parser
    if _parser is None:
        with _lock:
            if _parser is None:
                _parser = StatementParserLocal()
    return _parser


def reload_parser():
    global _parser
    with _lock:
        _parser = StatementParserLocal()
    return _parser


def main():
    parser = argparse.ArgumentParser(description="Parse a bank statement PDF")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"File not found: {args.pdf}")
        sys.exit(1)

    print("Loading model...")
    stmt_parser = get_parser()
    print(f"Model version: {stmt_parser.version}")

    print(f"Parsing {pdf_path.name}...")
    pdf_bytes = pdf_path.read_bytes()
    result = stmt_parser.parse_pdf(pdf_bytes)

    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
        return

    print(f"\nPages: {result.page_count}, Words: {result.word_count}")
    print(f"Transactions found: {len(result.transactions)}")
    print(f"Total deposits:     ${result.total_deposits:,.2f}")
    print(f"Total withdrawals:  ${result.total_withdrawals:,.2f}")
    if result.average_daily_balance:
        print(f"Avg daily balance:  ${result.average_daily_balance:,.2f}")
    print(f"NSF/overdraft fees: {result.nsf_count}")

    if result.low_confidence_pages:
        print(f"\nLow confidence pages: {result.low_confidence_pages}")

    print(f"\n{'DATE':<12} {'DESCRIPTION':<45} {'AMOUNT':>12} {'TYPE':<6} {'BALANCE':>12}")
    print("-" * 95)
    for t in result.transactions:
        bal = f"${t['balance_after']:,.2f}" if t.get("balance_after") is not None else ""
        amt = f"${t['amount']:,.2f}"
        print(f"{t['date']:<12} {t['description'][:43]:<45} {amt:>12} {t['type']:<6} {bal:>12}")


if __name__ == "__main__":
    main()
