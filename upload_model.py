#!/usr/bin/env python3
"""
Upload model weights to Supabase Storage for deployment.

Run this locally after training to make weights available to the deployed service.
The deployed container runs download_model.py at startup to fetch them.

Usage:
    python upload_model.py                    # upload both models
    python upload_model.py --parser-only      # just LayoutLMv3
    python upload_model.py --credit-debit-only # just FFN classifier
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

BUCKET_NAME = "ml-models"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")

# Statement parser files
PARSER_DIR = PROJECT_DIR / "models" / "statement_parser"
PARSER_PREFIX = "statement_parser/"
PARSER_FILES = [
    "config.json", "model.safetensors", "meta.json",
    "tokenizer_config.json", "tokenizer.json", "vocab.json", "vocab.txt",
    "merges.txt", "special_tokens_map.json", "preprocessor_config.json",
]

# Credit/debit classifier files
CREDIT_DEBIT_DIR = PROJECT_DIR / "models" / "credit_debit"
CREDIT_DEBIT_PREFIX = "credit_debit/"
CREDIT_DEBIT_FILES = ["model.pt", "tfidf.pkl", "meta.json"]


def _upload_files(sb, local_dir: Path, remote_prefix: str, files: list) -> int:
    """Upload model files to Supabase Storage. Returns count uploaded."""
    uploaded = 0
    for filename in files:
        local_path = local_dir / filename
        if not local_path.exists():
            print(f"  Skipping {filename} (not found)")
            continue

        remote_path = f"{remote_prefix}{filename}"
        data = local_path.read_bytes()
        size_mb = len(data) / (1024 * 1024)

        try:
            # Try upload first; if exists, update
            try:
                sb.storage.from_(BUCKET_NAME).upload(remote_path, data)
            except Exception:
                sb.storage.from_(BUCKET_NAME).update(remote_path, data)
            print(f"  Uploaded {filename} ({size_mb:.1f} MB)")
            uploaded += 1
        except Exception as e:
            print(f"  ERROR uploading {filename}: {e}")

    return uploaded


def main():
    parser = argparse.ArgumentParser(description="Upload model weights to Supabase Storage")
    parser.add_argument("--parser-only", action="store_true")
    parser.add_argument("--credit-debit-only", action="store_true")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE in .env")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Ensure bucket exists
    try:
        sb.storage.create_bucket(BUCKET_NAME, options={"public": False})
        print(f"Created bucket '{BUCKET_NAME}'")
    except Exception:
        pass  # Already exists

    total = 0

    if not args.credit_debit_only:
        print(f"\nUploading statement parser ({PARSER_DIR})...")
        if not PARSER_DIR.exists():
            print("  ERROR: No model at models/statement_parser/. Run train.py first.")
        else:
            total += _upload_files(sb, PARSER_DIR, PARSER_PREFIX, PARSER_FILES)

    if not args.parser_only:
        print(f"\nUploading credit/debit classifier ({CREDIT_DEBIT_DIR})...")
        if not CREDIT_DEBIT_DIR.exists():
            print("  ERROR: No model at models/credit_debit/. Run credit_debit_model.py first.")
        else:
            total += _upload_files(sb, CREDIT_DEBIT_DIR, CREDIT_DEBIT_PREFIX, CREDIT_DEBIT_FILES)

    print(f"\nDone. Uploaded {total} files to Supabase Storage (bucket: {BUCKET_NAME})")


if __name__ == "__main__":
    main()
