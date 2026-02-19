#!/usr/bin/env python3
"""
Fine-tune LayoutLMv3 for bank statement transaction extraction (BIO token classification).

Input:  BIO-tagged training data from align_labels.py
Output: Fine-tuned model saved to models/statement_parser/

Usage:
    python train.py                        # train with defaults
    python train.py --epochs 5             # custom epochs
    python train.py --batch-size 4         # custom batch size
    python train.py --test-split 0.15      # 15% test set
    python train.py --resume               # resume from last checkpoint
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_DIR / "data" / "training"
PAGES_DIR = PROJECT_DIR / "data" / "pages"
MODEL_DIR = PROJECT_DIR / "models" / "statement_parser"

# BIO tag scheme
BIO_TAGS = [
    "O",
    "B-DATE", "I-DATE",
    "B-DESC", "I-DESC",
    "B-AMOUNT", "I-AMOUNT",
    "B-BALANCE", "I-BALANCE",
]
TAG2ID = {tag: i for i, tag in enumerate(BIO_TAGS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}
NUM_LABELS = len(BIO_TAGS)

# LayoutLMv3 limits
MAX_SEQ_LEN = 512
IMAGE_SIZE = 224


class StatementPageDataset(Dataset):
    """Dataset of BIO-tagged bank statement pages for LayoutLMv3."""

    def __init__(self, samples: list, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        words = sample["words"]
        bboxes = sample["bboxes"]
        tags = sample["tags"]
        image_path = sample["image_path"]

        # Load page image
        image = Image.open(image_path).convert("RGB")

        # Truncate to MAX_SEQ_LEN (LayoutLMv3 will further tokenize)
        words = words[:MAX_SEQ_LEN]
        bboxes = bboxes[:MAX_SEQ_LEN]
        tags = tags[:MAX_SEQ_LEN]

        # Convert tags to IDs
        tag_ids = [TAG2ID.get(t, 0) for t in tags]

        # Process with LayoutLMv3Processor
        encoding = self.processor(
            image,
            words,
            boxes=bboxes,
            word_labels=tag_ids,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )

        # Squeeze batch dimension
        return {k: v.squeeze(0) for k, v in encoding.items()}


LABELS_DIR = PROJECT_DIR / "data" / "labels"


def load_training_data(min_alignment: float = 50.0, only_reviewed: bool = False,
                       claude_only: bool = False) -> list:
    """Load all BIO-tagged training samples.

    Args:
        claude_only: If True, only load pages that have Claude Vision labels
                     (skips rule-based labels from label_rules.py).
    """
    samples = []

    if not TRAINING_DIR.exists():
        print("No training data. Run align_labels.py first.")
        sys.exit(1)

    # Build set of Claude-labeled pages for fast lookup
    claude_pages = set()
    if claude_only and LABELS_DIR.exists():
        for label_dir in LABELS_DIR.iterdir():
            if not label_dir.is_dir():
                continue
            for lf in label_dir.glob("page_*_labels.json"):
                page_idx = int(lf.stem.replace("page_", "").replace("_labels", ""))
                claude_pages.add((label_dir.name, page_idx))
        print(f"Claude-labeled pages found: {len(claude_pages)}")

    skipped_rule_based = 0

    for pdf_dir in sorted(TRAINING_DIR.iterdir()):
        if not pdf_dir.is_dir():
            continue

        for bio_file in sorted(pdf_dir.glob("page_*_bio.json")):
            with open(bio_file) as f:
                data = json.load(f)

            page_idx = data["page_index"]
            pdf_stem = data["pdf_stem"]

            # Filter to Claude-labeled pages only
            if claude_only and (pdf_stem, page_idx) not in claude_pages:
                skipped_rule_based += 1
                continue

            # Filter by alignment quality
            if data.get("alignment_rate", 0) < min_alignment:
                continue

            # Filter by review status if requested
            if only_reviewed and not data.get("reviewed"):
                continue
            if only_reviewed and data.get("review_status") == "rejected":
                continue

            # Need corresponding page image
            image_path = PAGES_DIR / pdf_stem / f"page_{page_idx}.png"
            if not image_path.exists():
                continue

            if not data.get("words") or not data.get("tags"):
                continue

            samples.append({
                "words": data["words"],
                "bboxes": data["bboxes"],
                "tags": data["tags"],
                "image_path": str(image_path),
                "pdf_stem": pdf_stem,
                "page_index": page_idx,
            })

    if claude_only:
        print(f"Skipped {skipped_rule_based} rule-based pages")

    return samples


def train(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    test_split: float = 0.15,
    min_alignment: float = 80.0,
    only_reviewed: bool = False,
    claude_only: bool = False,
    resume: bool = False,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading training data...")
    samples = load_training_data(min_alignment=min_alignment, only_reviewed=only_reviewed,
                                    claude_only=claude_only)
    if len(samples) < 10:
        print(f"Only {len(samples)} samples found. Need at least 10 to train.")
        sys.exit(1)

    # Tag distribution + class weights
    all_tags = []
    for s in samples:
        all_tags.extend(s["tags"])
    tag_dist = Counter(all_tags)
    total_tags = sum(tag_dist.values())
    print(f"\nLoaded {len(samples)} pages")
    print(f"Tag distribution: {dict(tag_dist)}")

    # Inverse-frequency class weights (capped to avoid extreme values)
    class_weights = torch.ones(NUM_LABELS, device=device)
    for tag, count in tag_dist.items():
        tid = TAG2ID.get(tag)
        if tid is not None and count > 0:
            weight = total_tags / (NUM_LABELS * count)
            class_weights[tid] = min(weight, 10.0)  # cap at 10x
    print(f"Class weights: {dict(zip(BIO_TAGS, [round(w.item(), 2) for w in class_weights]))}")

    # Train/test split — group by PDF to prevent data leakage
    # (pages from the same statement must stay in the same split)
    pdf_stems = sorted(set(s["pdf_stem"] for s in samples))
    train_stems, test_stems = train_test_split(
        pdf_stems, test_size=test_split, random_state=42
    )
    test_stem_set = set(test_stems)
    train_samples = [s for s in samples if s["pdf_stem"] not in test_stem_set]
    test_samples = [s for s in samples if s["pdf_stem"] in test_stem_set]
    print(f"Train: {len(train_samples)} pages ({len(train_stems)} PDFs), "
          f"Test: {len(test_samples)} pages ({len(test_stems)} PDFs)")

    # Save test split for evaluation
    test_manifest = [{"pdf_stem": s["pdf_stem"], "page_index": s["page_index"]} for s in test_samples]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "test_split.json", "w") as f:
        json.dump(test_manifest, f, indent=2)

    # Load model and processor
    print("\nLoading LayoutLMv3...")
    if resume and (MODEL_DIR / "config.json").exists():
        print("  Resuming from local checkpoint")
        model = LayoutLMv3ForTokenClassification.from_pretrained(str(MODEL_DIR))
        processor = LayoutLMv3Processor.from_pretrained(str(MODEL_DIR))
    else:
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=NUM_LABELS,
            id2label=ID2TAG,
            label2id=TAG2ID,
        )
        processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False,  # We provide our own words + bboxes
        )

    model.to(device)

    # Datasets
    train_dataset = StatementPageDataset(train_samples, processor)
    test_dataset = StatementPageDataset(test_samples, processor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_f1 = 0.0
    best_report = None
    print(f"\nTraining for {epochs} epochs ({total_steps} steps, {warmup_steps} warmup)...\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            # Weighted cross-entropy to handle class imbalance
            logits = outputs.logits  # (batch, seq_len, num_labels)
            labels = batch["labels"]  # (batch, seq_len)
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            loss = loss_fn(logits.view(-1, NUM_LABELS), labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1

            if (batch_idx + 1) % 20 == 0:
                avg_loss = epoch_loss / epoch_steps
                print(f"  Epoch {epoch+1}/{epochs} | Step {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

        avg_loss = epoch_loss / epoch_steps
        print(f"\nEpoch {epoch+1}/{epochs} — Avg loss: {avg_loss:.4f}")

        # Evaluate
        model.eval()
        all_preds = []  # list of sequences (one per page)
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)

                labels_batch = batch["labels"]
                for pred_seq, label_seq in zip(predictions, labels_batch):
                    page_preds = []
                    page_labels = []
                    for p, l in zip(pred_seq, label_seq):
                        if l.item() != -100:  # ignore padding
                            page_preds.append(ID2TAG[p.item()])
                            page_labels.append(ID2TAG[l.item()])
                    if page_labels:
                        all_preds.append(page_preds)
                        all_labels.append(page_labels)

        # Calculate entity-level metrics (seqeval expects list of sequences)
        from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)

        print(f"  Entity F1: {f1:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_report = classification_report(all_labels, all_preds, output_dict=True)
            print(f"  New best F1! Saving model...")

            model.save_pretrained(str(MODEL_DIR))
            processor.save_pretrained(str(MODEL_DIR))

            # Print per-tag breakdown
            report_str = classification_report(all_labels, all_preds)
            print(report_str)

    # Save metadata with per-tag evaluation
    per_tag_metrics = {}
    if best_report:
        for tag in BIO_TAGS:
            if tag != "O" and tag in best_report:
                per_tag_metrics[tag] = {
                    "precision": round(best_report[tag]["precision"], 4),
                    "recall": round(best_report[tag]["recall"], 4),
                    "f1": round(best_report[tag]["f1-score"], 4),
                    "support": best_report[tag]["support"],
                }

    meta = {
        "version": f"v{int(time.time())}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "base_model": "microsoft/layoutlmv3-base",
        "num_labels": NUM_LABELS,
        "bio_tags": BIO_TAGS,
        "tag2id": TAG2ID,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train_pdfs": len(train_stems),
        "test_pdfs": len(test_stems),
        "min_alignment": min_alignment,
        "claude_only": claude_only,
        "evaluation": {
            "best_f1": round(best_f1, 4),
            "per_tag": per_tag_metrics,
        },
        "class_weights": {BIO_TAGS[i]: round(class_weights[i].item(), 2) for i in range(NUM_LABELS)},
        "tag_distribution": dict(tag_dist),
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 for statement parsing")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--min-alignment", type=float, default=80.0,
                        help="Minimum alignment rate to include page in training")
    parser.add_argument("--only-reviewed", action="store_true",
                        help="Only use human-reviewed pages")
    parser.add_argument("--claude-only", action="store_true",
                        help="Only use Claude Vision labeled pages (skip rule-based)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        test_split=args.test_split,
        min_alignment=args.min_alignment,
        only_reviewed=args.only_reviewed,
        claude_only=args.claude_only,
        resume=args.resume,
    )
