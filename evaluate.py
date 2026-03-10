#!/usr/bin/env python3
"""
Evaluate the trained LayoutLMv3 model at two levels:
  1. Token-level: precision/recall/F1 per BIO tag (using seqeval)
  2. Transaction-level: match predicted vs ground truth transactions

Uses the same chunked inference as predict.py to evaluate ALL words on
dense pages (not just the first 512 sub-tokens).

Targets:
  - Token F1 > 0.90
  - Transaction precision > 0.95
  - Transaction recall > 0.90

Usage:
    python evaluate.py                    # evaluate on test split
    python evaluate.py --all              # evaluate on all data
    python evaluate.py --verbose          # show per-page details
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from rapidfuzz import fuzz

PROJECT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_DIR / "data" / "training"
PAGES_DIR = PROJECT_DIR / "data" / "pages"
LABELS_DIR = PROJECT_DIR / "data" / "labels"
MODEL_DIR = PROJECT_DIR / "models" / "statement_parser"

MAX_SEQ_LEN = 512

# Load tag scheme from meta.json or use defaults
BIO_TAGS = ["O", "B-DATE", "I-DATE", "B-DESC", "I-DESC", "B-AMOUNT", "I-AMOUNT", "B-BALANCE", "I-BALANCE"]
TAG2ID = {tag: i for i, tag in enumerate(BIO_TAGS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}

# Bank detection patterns (shared with predict.py)
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


def _detect_bank(words: list) -> str:
    """Detect bank from first 100 words of a page."""
    text = " ".join(words[:100]).lower()
    for pattern, bank in _BANK_PATTERNS.items():
        if pattern in text:
            return bank
    return "unknown"


def _load_test_pages(use_all: bool = False) -> list:
    """Load test set pages."""
    if not use_all:
        split_path = MODEL_DIR / "test_split.json"
        if split_path.exists():
            with open(split_path) as f:
                manifest = json.load(f)
            return manifest

    # Use all pages
    pages = []
    for pdf_dir in sorted(TRAINING_DIR.iterdir()):
        if not pdf_dir.is_dir():
            continue
        for bio_file in sorted(pdf_dir.glob("page_*_bio.json")):
            with open(bio_file) as f:
                data = json.load(f)
            pages.append({"pdf_stem": data["pdf_stem"], "page_index": data["page_index"]})
    return pages


def _bio_tags_to_transactions(words: list, tags: list) -> list:
    """Convert BIO-tagged word sequence to structured transactions.

    Uses the same field-specific join logic as predict.py:
    - amounts/balances: no space (e.g. "1,234" + ".56" → "1,234.56")
    - dates: conditional spacing (no space after "/" or "-")
    - descriptions: always space-separated
    """
    transactions = []
    current = {}

    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            field = tag[2:].lower()
            if field == "date" and current and "date" in current:
                transactions.append(current.copy())
                current = {}
            current[field] = word

        elif tag.startswith("I-"):
            field = tag[2:].lower()
            if field in current:
                # Field-specific join separators (matches predict.py)
                if field in ("amount", "balance"):
                    sep = ""
                elif field == "date":
                    prev = current[field]
                    sep = "" if prev.endswith(("/", "-")) or word in ("/", "-", ",") else " "
                else:
                    sep = " "
                current[field] += f"{sep}{word}"
            elif current:
                # Orphan I-tag without B-tag: promote to B-tag
                current[field] = word

    if current and "date" in current:
        transactions.append(current)

    return transactions


def _parse_amount(text: str) -> float:
    """Parse amount string to float."""
    if not text:
        return 0.0
    cleaned = re.sub(r"[,$()\s]", "", text)
    cleaned = cleaned.replace("−", "-").replace("–", "-")
    # Handle trailing minus sign (e.g., "12.00-" → "-12.00")
    if cleaned.endswith("-"):
        cleaned = "-" + cleaned[:-1]
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _match_transactions(predicted: list, ground_truth: list) -> dict:
    """Match predicted transactions against ground truth.

    A prediction matches if:
      - Date tokens overlap (substring containment)
      - Amount within ±$0.01
    When multiple predictions match the same GT, uses description
    similarity as a tiebreaker.
    """
    if not predicted or not ground_truth:
        return {
            "tp": 0,
            "fp": len(predicted),
            "fn": len(ground_truth),
        }

    matched_pred = set()
    matched_gt = set()

    for gi, gt_txn in enumerate(ground_truth):
        gt_date = gt_txn.get("date", "")
        gt_amount = gt_txn.get("amount", 0)
        gt_desc = gt_txn.get("description", "")
        if isinstance(gt_amount, str):
            gt_amount = _parse_amount(gt_amount)

        # Find all candidate matches for this GT transaction
        candidates = []
        for pi, pred_txn in enumerate(predicted):
            if pi in matched_pred:
                continue

            pred_date = pred_txn.get("date", "")
            pred_amount_str = pred_txn.get("amount", "0")
            pred_amount = _parse_amount(pred_amount_str) if isinstance(pred_amount_str, str) else pred_amount_str

            # Date match: substring containment
            date_match = (
                gt_date.lower() in pred_date.lower()
                or pred_date.lower() in gt_date.lower()
            )

            # Amount match: within $0.01
            amount_match = abs(abs(pred_amount) - abs(gt_amount)) <= 0.01

            if date_match and amount_match:
                # Compute description similarity as tiebreaker
                pred_desc = pred_txn.get("desc", pred_txn.get("description", ""))
                desc_sim = fuzz.ratio(gt_desc.lower(), pred_desc.lower()) if gt_desc and pred_desc else 0
                candidates.append((pi, desc_sim))

        # Pick the best candidate (highest description similarity)
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_pi = candidates[0][0]
            matched_pred.add(best_pi)
            matched_gt.add(gi)

    tp = len(matched_gt)
    fp = len(predicted) - len(matched_pred)
    fn = len(ground_truth) - len(matched_gt)

    return {"tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Chunked inference (mirrors predict.py._predict_chunk / _predict_page)
# ---------------------------------------------------------------------------

def _predict_chunk(model, processor, device, word_texts, word_bboxes, image):
    """Run inference on a single chunk. Returns (tags, confidences)."""
    encoding = processor(
        image, word_texts, boxes=word_bboxes,
        truncation=True, max_length=MAX_SEQ_LEN, padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        device_encoding = {k: v.to(device) for k, v in encoding.items()}
        outputs = model(**device_encoding)
        logits = outputs.logits.squeeze(0).cpu()
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        max_probs = probs.max(dim=-1).values

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


def _predict_page(model, processor, device, words, bboxes, image):
    """Run inference with automatic chunking for dense pages (mirrors predict.py)."""
    if not words:
        return [], []

    # First pass: try all words at once
    chunk_texts = words[:MAX_SEQ_LEN]
    chunk_bboxes = bboxes[:MAX_SEQ_LEN]

    tags, confidences = _predict_chunk(model, processor, device, chunk_texts, chunk_bboxes, image)

    # Check how many words actually fit
    test_enc = processor(
        image, chunk_texts, boxes=chunk_bboxes,
        truncation=True, max_length=MAX_SEQ_LEN, padding="max_length",
        return_tensors="pt",
    )
    word_ids = test_enc.word_ids(batch_index=0)
    mapped_ids = [wid for wid in word_ids if wid is not None]
    max_mapped = max(mapped_ids) if mapped_ids else 0

    if max_mapped >= len(words) - 1:
        return tags, confidences

    # Dense page: chunked inference
    words_per_chunk = max(max_mapped - 10, 50)
    overlap = 20

    all_tags = ["O"] * len(words)
    all_confs = [0.0] * len(words)

    for i in range(min(max_mapped + 1, len(all_tags))):
        all_tags[i] = tags[i]
        all_confs[i] = confidences[i]

    global_frontier = max_mapped
    start = max(global_frontier + 1 - overlap, 0)

    for _ in range(20):
        if start >= len(words):
            break

        end = min(start + words_per_chunk, len(words))
        chunk_t = words[start:end]
        chunk_b = bboxes[start:end]

        if not chunk_t:
            break

        c_tags, c_confs = _predict_chunk(model, processor, device, chunk_t, chunk_b, image)

        chunk_last_mapped = -1
        for i in range(len(c_confs) - 1, -1, -1):
            if c_confs[i] > 0:
                chunk_last_mapped = i
                break

        if chunk_last_mapped < 0:
            break

        for i in range(chunk_last_mapped + 1):
            global_i = start + i
            if global_i >= len(all_tags):
                break
            if c_confs[i] > all_confs[global_i]:
                all_tags[global_i] = c_tags[i]
                all_confs[global_i] = c_confs[i]

        new_frontier = start + chunk_last_mapped
        if new_frontier <= global_frontier:
            break
        global_frontier = new_frontier

        if global_frontier >= len(words) - 1:
            break

        start = max(global_frontier + 1 - overlap, start + 1)

    return all_tags, all_confs


# ---------------------------------------------------------------------------
# Token-level evaluation with chunked inference
# ---------------------------------------------------------------------------

def _evaluate_page_tokens(model, processor, device, words, bboxes, true_tags, image):
    """Run chunked inference and return aligned (true, pred) tag lists.

    Only returns tags for words that were in the ground truth (handles the
    case where ground truth was truncated at MAX_SEQ_LEN during training).
    """
    # Get predictions for ALL words via chunked inference
    pred_tags, _ = _predict_page(model, processor, device, words, bboxes, image)

    # Align: only evaluate words that have ground truth tags
    n = min(len(true_tags), len(pred_tags))
    page_true = true_tags[:n]
    page_pred = pred_tags[:n]

    return page_true, page_pred


def evaluate(use_all: bool = False, verbose: bool = False):
    """Run full evaluation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    if not (MODEL_DIR / "config.json").exists():
        print(f"No model found at {MODEL_DIR}. Run train.py first.")
        sys.exit(1)

    print(f"Loading model (device: {device})...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(MODEL_DIR))
    processor = LayoutLMv3Processor.from_pretrained(str(MODEL_DIR), apply_ocr=False)
    model.to(device)
    model.eval()

    # Load test pages
    test_pages = _load_test_pages(use_all=use_all)
    print(f"Evaluating on {len(test_pages)} pages...")

    all_true_tags = []
    all_pred_tags = []
    txn_tp = 0
    txn_fp = 0
    txn_fn = 0
    truncated_pages = 0
    bank_metrics = {}  # bank -> {tp, fp, fn}

    for page_info in test_pages:
        stem = page_info["pdf_stem"]
        page_idx = page_info["page_index"]

        bio_path = TRAINING_DIR / stem / f"page_{page_idx}_bio.json"
        image_path = PAGES_DIR / stem / f"page_{page_idx}.png"

        if not bio_path.exists() or not image_path.exists():
            continue

        with open(bio_path) as f:
            bio_data = json.load(f)

        words = bio_data["words"]
        bboxes = bio_data["bboxes"]
        true_tags = bio_data["tags"]

        image = Image.open(image_path).convert("RGB")

        # Track if page has more words than can fit in single pass
        if len(words) > 300:
            truncated_pages += 1

        # Token-level: chunked inference on ALL words
        page_true, page_pred = _evaluate_page_tokens(
            model, processor, device, words, bboxes, true_tags, image
        )

        if page_true:
            all_true_tags.append(page_true)
            all_pred_tags.append(page_pred)

        # Transaction-level: use predictions for ALL words
        pred_tags_full, _ = _predict_page(model, processor, device, words, bboxes, image)
        pred_txns = _bio_tags_to_transactions(words[:len(pred_tags_full)], pred_tags_full)

        # Load Claude's ground truth transactions (skip parse-error pages)
        label_path = LABELS_DIR / stem / f"page_{page_idx}_labels.json"
        gt_txns = []
        is_parse_error = False
        if label_path.exists():
            with open(label_path) as f:
                label_data = json.load(f)
            gt_txns = label_data.get("transactions", [])
            is_parse_error = label_data.get("parse_error", False)

        # Count FPs even when gt is empty (unless it's a parse error — bad ground truth)
        if not gt_txns and pred_txns and not is_parse_error:
            # Page has no ground truth txns and model predicts some = all FP
            match_result = {"tp": 0, "fp": len(pred_txns), "fn": 0}
            txn_fp += match_result["fp"]
            bank = _detect_bank(words)
            if bank not in bank_metrics:
                bank_metrics[bank] = {"tp": 0, "fp": 0, "fn": 0}
            bank_metrics[bank]["fp"] += match_result["fp"]

        if gt_txns:
            match_result = _match_transactions(pred_txns, gt_txns)
            txn_tp += match_result["tp"]
            txn_fp += match_result["fp"]
            txn_fn += match_result["fn"]

            # Per-bank metrics
            bank = _detect_bank(words)
            if bank not in bank_metrics:
                bank_metrics[bank] = {"tp": 0, "fp": 0, "fn": 0}
            bank_metrics[bank]["tp"] += match_result["tp"]
            bank_metrics[bank]["fp"] += match_result["fp"]
            bank_metrics[bank]["fn"] += match_result["fn"]

        if verbose:
            page_f1 = f1_score([page_true], [page_pred]) if page_true else 0
            print(f"  {stem}/page_{page_idx}: F1={page_f1:.3f} "
                  f"pred_txns={len(pred_txns)} gt_txns={len(gt_txns)} words={len(words)}")

    # Token-level metrics
    print(f"\n{'='*60}")
    print("TOKEN-LEVEL METRICS (seqeval entity-level)")
    print(f"{'='*60}")

    token_f1 = f1_score(all_true_tags, all_pred_tags)
    token_precision = precision_score(all_true_tags, all_pred_tags)
    token_recall = recall_score(all_true_tags, all_pred_tags)

    print(f"  F1:        {token_f1:.4f}  {'PASS' if token_f1 > 0.90 else 'BELOW TARGET (>0.90)'}")
    print(f"  Precision: {token_precision:.4f}")
    print(f"  Recall:    {token_recall:.4f}")
    if truncated_pages:
        print(f"  (Chunked inference used on {truncated_pages} dense pages)")
    print(f"\nDetailed report:")
    print(classification_report(all_true_tags, all_pred_tags))

    # Transaction-level metrics
    print(f"\n{'='*60}")
    print("TRANSACTION-LEVEL METRICS")
    print(f"{'='*60}")

    txn_precision = txn_tp / (txn_tp + txn_fp) if (txn_tp + txn_fp) > 0 else 0
    txn_recall = txn_tp / (txn_tp + txn_fn) if (txn_tp + txn_fn) > 0 else 0
    txn_f1 = 2 * txn_precision * txn_recall / (txn_precision + txn_recall) if (txn_precision + txn_recall) > 0 else 0

    print(f"  True positives:  {txn_tp}")
    print(f"  False positives: {txn_fp}")
    print(f"  False negatives: {txn_fn}")
    print(f"  Precision: {txn_precision:.4f}  {'PASS' if txn_precision > 0.95 else 'BELOW TARGET (>0.95)'}")
    print(f"  Recall:    {txn_recall:.4f}  {'PASS' if txn_recall > 0.90 else 'BELOW TARGET (>0.90)'}")
    print(f"  F1:        {txn_f1:.4f}")

    # Per-bank metrics
    if bank_metrics:
        print(f"\n{'='*60}")
        print("PER-BANK TRANSACTION METRICS")
        print(f"{'='*60}")
        print(f"  {'Bank':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Recall':>7} {'F1':>7}")
        print(f"  {'-'*56}")

        bank_results = {}
        for bank, m in sorted(bank_metrics.items(), key=lambda x: -(x[1]["tp"] + x[1]["fn"])):
            bp = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
            br = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
            bf1 = 2 * bp * br / (bp + br) if (bp + br) > 0 else 0
            print(f"  {bank:<20} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {bp:>7.3f} {br:>7.3f} {bf1:>7.3f}")
            bank_results[bank] = {
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "precision": round(bp, 4), "recall": round(br, 4), "f1": round(bf1, 4),
            }

    # Save evaluation results
    results = {
        "pages_evaluated": len(test_pages),
        "dense_pages_chunked": truncated_pages,
        "token_metrics": {
            "f1": round(token_f1, 4),
            "precision": round(token_precision, 4),
            "recall": round(token_recall, 4),
        },
        "transaction_metrics": {
            "tp": txn_tp,
            "fp": txn_fp,
            "fn": txn_fn,
            "precision": round(txn_precision, 4),
            "recall": round(txn_recall, 4),
            "f1": round(txn_f1, 4),
        },
        "per_bank": bank_results if bank_metrics else {},
    }

    eval_path = MODEL_DIR / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate statement parser model")
    parser.add_argument("--all", action="store_true", help="Evaluate on all data, not just test split")
    parser.add_argument("--verbose", action="store_true", help="Show per-page details")
    args = parser.parse_args()

    evaluate(use_all=args.all, verbose=args.verbose)
