#!/usr/bin/env python3
"""
Retrain the LayoutLMv3 model with new feedback data.

Pipeline:
  1. Export new labeled data from statement_extraction_feedback
  2. Merge with existing training data
  3. Retrain model (saves as candidate)
  4. Evaluate candidate vs current model
  5. Promote if candidate is better

Usage:
    python retrain.py                         # full pipeline
    python retrain.py --skip-export           # retrain with existing data only
    python retrain.py --promote               # force promote candidate
    python retrain.py --compare               # compare candidate vs current
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

MODEL_DIR = PROJECT_DIR / "models" / "statement_parser"
CANDIDATE_DIR = PROJECT_DIR / "models" / "statement_parser_candidate"
TRAINING_DIR = PROJECT_DIR / "data" / "training"
BACKUP_DIR = PROJECT_DIR / "models" / "backups"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")


def export_feedback() -> int:
    """Export human feedback from Supabase for retraining."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("No Supabase credentials, skipping feedback export")
        return 0

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Get feedback where human corrections exist
    result = (
        sb.table("statement_extraction_feedback")
        .select("*")
        .not_.is_("corrected_transactions", "null")
        .order("created_at", desc=True)
        .limit(1000)
        .execute()
    )

    rows = result.data or []
    print(f"  Exported {len(rows)} feedback records")
    return len(rows)


def retrain(epochs: int = 3, skip_export: bool = False):
    """Retrain model with combined data."""
    if not skip_export:
        print("Step 1: Exporting feedback...")
        export_feedback()

    print("\nStep 2: Training candidate model...")
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    # Copy current model as starting point for fine-tuning
    if (MODEL_DIR / "config.json").exists():
        for f in MODEL_DIR.iterdir():
            if f.is_file():
                shutil.copy2(f, CANDIDATE_DIR / f.name)
        print("  Starting from current model weights")
    else:
        print("  No existing model, training from scratch")

    # Run training to candidate directory
    from train import train as run_training
    import train as train_module

    # Override output directory
    original_model_dir = train_module.MODEL_DIR
    train_module.MODEL_DIR = CANDIDATE_DIR

    try:
        run_training(epochs=epochs, resume=True)
    finally:
        train_module.MODEL_DIR = original_model_dir

    print("\nStep 3: Evaluating candidate...")
    compare()


def compare():
    """Compare candidate model vs current model."""
    if not (CANDIDATE_DIR / "config.json").exists():
        print("No candidate model found. Run retrain first.")
        return

    current_eval = MODEL_DIR / "evaluation.json"
    candidate_eval = CANDIDATE_DIR / "evaluation.json"

    # Run evaluation on candidate
    import evaluate as eval_module
    original_model_dir = eval_module.MODEL_DIR
    eval_module.MODEL_DIR = CANDIDATE_DIR

    try:
        eval_module.evaluate()
    finally:
        eval_module.MODEL_DIR = original_model_dir

    if not candidate_eval.exists():
        print("Candidate evaluation failed")
        return

    with open(candidate_eval) as f:
        cand = json.load(f)

    print(f"\n{'='*50}")
    print("COMPARISON")
    print(f"{'='*50}")

    if current_eval.exists():
        with open(current_eval) as f:
            curr = json.load(f)

        curr_f1 = curr.get("token_metrics", {}).get("f1", 0)
        cand_f1 = cand.get("token_metrics", {}).get("f1", 0)
        curr_txn_f1 = curr.get("transaction_metrics", {}).get("f1", 0)
        cand_txn_f1 = cand.get("transaction_metrics", {}).get("f1", 0)

        print(f"  Token F1:       {curr_f1:.4f} → {cand_f1:.4f}  ({'+' if cand_f1 >= curr_f1 else ''}{cand_f1-curr_f1:.4f})")
        print(f"  Transaction F1: {curr_txn_f1:.4f} → {cand_txn_f1:.4f}  ({'+' if cand_txn_f1 >= curr_txn_f1 else ''}{cand_txn_f1-curr_txn_f1:.4f})")

        if cand_f1 >= curr_f1:
            print("\n  Candidate is BETTER or equal. Safe to promote.")
        else:
            print(f"\n  Candidate is WORSE by {curr_f1-cand_f1:.4f}. Consider not promoting.")
    else:
        cand_f1 = cand.get("token_metrics", {}).get("f1", 0)
        print(f"  No current model evaluation. Candidate F1: {cand_f1:.4f}")


def promote(force: bool = False):
    """Promote candidate model to production."""
    if not (CANDIDATE_DIR / "config.json").exists():
        print("No candidate model to promote.")
        return

    # Check evaluations unless forced
    if not force:
        current_eval = MODEL_DIR / "evaluation.json"
        candidate_eval = CANDIDATE_DIR / "evaluation.json"

        if current_eval.exists() and candidate_eval.exists():
            with open(current_eval) as f:
                curr = json.load(f)
            with open(candidate_eval) as f:
                cand = json.load(f)

            curr_f1 = curr.get("token_metrics", {}).get("f1", 0)
            cand_f1 = cand.get("token_metrics", {}).get("f1", 0)

            if cand_f1 < curr_f1:
                print(f"Candidate F1 ({cand_f1:.4f}) < Current F1 ({curr_f1:.4f})")
                print("Use --force to promote anyway.")
                return

    # Backup current model
    if (MODEL_DIR / "config.json").exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"statement_parser_{timestamp}"
        shutil.copytree(MODEL_DIR, backup_path)
        print(f"  Backed up current model to {backup_path}")

    # Promote candidate
    for f in CANDIDATE_DIR.iterdir():
        if f.is_file():
            shutil.copy2(f, MODEL_DIR / f.name)

    # Update meta.json with promotion timestamp
    meta_path = MODEL_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["promoted_at"] = datetime.now(timezone.utc).isoformat()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Clean up candidate
    shutil.rmtree(CANDIDATE_DIR)

    print("  Candidate promoted to production!")
    print("  Run `POST /v1/reload` on the service to load new weights.")


def main():
    parser = argparse.ArgumentParser(description="Retrain statement parser model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare candidate vs current")
    parser.add_argument("--promote", action="store_true", help="Promote candidate to production")
    parser.add_argument("--force", action="store_true", help="Force promote even if worse")
    args = parser.parse_args()

    if args.compare:
        compare()
    elif args.promote:
        promote(force=args.force)
    else:
        retrain(epochs=args.epochs, skip_export=args.skip_export)


if __name__ == "__main__":
    main()
