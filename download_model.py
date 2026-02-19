#!/usr/bin/env python3
"""
Download model weights from Supabase Storage at service startup.

Checks if local models exist; if not, downloads from the 'ml-models' bucket.
Downloads both the LayoutLMv3 statement parser and the credit/debit FFN classifier.

Usage:
    python download_model.py                    # download if missing
    python download_model.py --force            # force re-download
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

BUCKET_NAME = "ml-models"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")

# --- LayoutLMv3 Statement Parser ---
PARSER_DIR = PROJECT_DIR / "models" / "statement_parser"
PARSER_PREFIX = "statement_parser/"
PARSER_REQUIRED = ["config.json", "model.safetensors", "meta.json"]
PARSER_OPTIONAL = [
    "tokenizer_config.json", "tokenizer.json", "vocab.json", "vocab.txt",
    "merges.txt", "special_tokens_map.json", "preprocessor_config.json",
]

# --- Credit/Debit FFN Classifier ---
CREDIT_DEBIT_DIR = PROJECT_DIR / "models" / "credit_debit"
CREDIT_DEBIT_PREFIX = "credit_debit/"
CREDIT_DEBIT_REQUIRED = ["model.pt", "tfidf.pkl", "meta.json"]


def _download_files(sb, local_dir: Path, remote_prefix: str,
                    required: list, optional: list, force: bool) -> bool:
    """Download a set of model files from Supabase Storage."""
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for filename in required + optional:
        remote_path = f"{remote_prefix}{filename}"
        local_path = local_dir / filename

        if not force and local_path.exists():
            continue

        try:
            print(f"  Downloading {remote_prefix}{filename}...")
            data = sb.storage.from_(BUCKET_NAME).download(remote_path)
            local_path.write_bytes(data)
            size_mb = len(data) / (1024 * 1024)
            print(f"    {size_mb:.1f} MB")
            downloaded += 1
        except Exception as e:
            if filename in required:
                print(f"  ERROR: Failed to download required file {filename}: {e}")
                return False
            else:
                print(f"  Skipping optional {filename}: {e}")

    if downloaded:
        print(f"  Downloaded {downloaded} files to {local_dir}")
    return True


def download_model(force: bool = False):
    # Check if statement parser already exists
    parser_exists = all((PARSER_DIR / f).exists() for f in PARSER_REQUIRED)
    cd_exists = all((CREDIT_DEBIT_DIR / f).exists() for f in CREDIT_DEBIT_REQUIRED)

    if not force and parser_exists and cd_exists:
        print(f"All models already exist")
        meta_path = PARSER_DIR / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                version = json.load(f).get("version", "?")
            print(f"  Statement parser version: {version}")
        return True

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("SUPABASE_URL and SUPABASE_SERVICE_ROLE required for model download")
        # Not fatal — service will start in degraded mode (Claude Vision only)
        return False

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Download statement parser
    if force or not parser_exists:
        print("Downloading statement parser model...")
        ok = _download_files(sb, PARSER_DIR, PARSER_PREFIX,
                             PARSER_REQUIRED, PARSER_OPTIONAL, force)
        if not ok:
            return False
    else:
        print("Statement parser model: up to date")

    # Download credit/debit classifier
    if force or not cd_exists:
        print("Downloading credit/debit classifier...")
        ok = _download_files(sb, CREDIT_DEBIT_DIR, CREDIT_DEBIT_PREFIX,
                             CREDIT_DEBIT_REQUIRED, [], force)
        if not ok:
            print("  Warning: credit/debit model not available, will use keyword fallback")
    else:
        print("Credit/debit classifier: up to date")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download statement parser models")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    success = download_model(force=args.force)
    sys.exit(0 if success else 1)
