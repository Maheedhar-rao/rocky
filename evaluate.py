#!/usr/bin/env python3
"""
Evaluate the trained LayoutLMv3 model at two levels:
  1. Token-level: precision/recall/F1 per BIO tag (using seqeval)
  2. Transaction-level: match predicted vs ground truth transactions

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
    """Convert BIO-tagged word sequence to structured transactions."""
    transactions = []
    current = {}

    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            # New entity start — if we hit B-DATE and have a pending txn, flush it
            field = tag[2:].lower()
            if field == "date" and current and "date" in current:
                transactions.append(current.copy())
                current = {}
            current[field] = word

        elif tag.startswith("I-"):
            field = tag[2:].lower()
            if field in current:
                current[field] += f" {word}"

    if current and "date" in current:
        transactions.append(current)

    return transactions


def _parse_amount(text: str) -> float:
    """Parse amount string to float."""
    if not text:
        return 0.0
    cleaned = re.sub(r"[,$()\\s]", "", text)
    cleaned = cleaned.replace("−", "-").replace("–", "-")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _match_transactions(predicted: list, ground_truth: list) -> dict:
    """Match predicted transactions against ground truth.

    A prediction matches if:
      - Date tokens overlap (exact substring match)
      - Amount within ±$0.01
    """
    matched_pred = set()
    matched_gt = set()

    for gi, gt_txn in enumerate(ground_truth):
        gt_date = gt_txn.get("date", "")
        gt_amount = gt_txn.get("amount", 0)
        if isinstance(gt_amount, str):
            gt_amount = _parse_amount(gt_amount)

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
                matched_pred.add(pi)
                matched_gt.add(gi)
                break

    tp = len(matched_gt)
    fp = len(predicted) - len(matched_pred)
    fn = len(ground_truth) - len(matched_gt)

    return {"tp": tp, "fp": fp, "fn": fn}


def evaluate(use_all: bool = False, verbose: bool = False):
    """Run full evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if not (MODEL_DIR / "config.json").exists():
        print(f"No model found at {MODEL_DIR}. Run train.py first.")
        sys.exit(1)

    print("Loading model...")
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

    for page_info in test_pages:
        stem = page_info["pdf_stem"]
        page_idx = page_info["page_index"]

        bio_path = TRAINING_DIR / stem / f"page_{page_idx}_bio.json"
        image_path = PAGES_DIR / stem / f"page_{page_idx}.png"

        if not bio_path.exists() or not image_path.exists():
            continue

        with open(bio_path) as f:
            bio_data = json.load(f)

        words = bio_data["words"][:MAX_SEQ_LEN]
        bboxes = bio_data["bboxes"][:MAX_SEQ_LEN]
        true_tags = bio_data["tags"][:MAX_SEQ_LEN]
        true_tag_ids = [TAG2ID.get(t, 0) for t in true_tags]

        image = Image.open(image_path).convert("RGB")

        # Run inference
        encoding = processor(
            image, words, boxes=bboxes, word_labels=true_tag_ids,
            truncation=True, max_length=MAX_SEQ_LEN, padding="max_length",
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Extract non-padding predictions
        labels = encoding["labels"].squeeze(0)
        preds = predictions.squeeze(0)

        page_true = []
        page_pred = []
        for p, l in zip(preds, labels):
            if l.item() != -100:
                page_true.append(ID2TAG[l.item()])
                page_pred.append(ID2TAG[p.item()])

        all_true_tags.append(page_true)
        all_pred_tags.append(page_pred)

        # Transaction-level evaluation
        pred_txns = _bio_tags_to_transactions(words[:len(page_pred)], page_pred)

        # Load Claude's ground truth transactions
        label_path = LABELS_DIR / stem / f"page_{page_idx}_labels.json"
        gt_txns = []
        if label_path.exists():
            with open(label_path) as f:
                gt_txns = json.load(f).get("transactions", [])

        if gt_txns:
            match_result = _match_transactions(pred_txns, gt_txns)
            txn_tp += match_result["tp"]
            txn_fp += match_result["fp"]
            txn_fn += match_result["fn"]

        if verbose:
            page_f1 = f1_score([page_true], [page_pred])
            print(f"  {stem}/page_{page_idx}: F1={page_f1:.3f} "
                  f"pred_txns={len(pred_txns)} gt_txns={len(gt_txns)}")

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

    # Save evaluation results
    results = {
        "pages_evaluated": len(test_pages),
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
