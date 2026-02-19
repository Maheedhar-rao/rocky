#!/usr/bin/env python3
"""
Download bank statement PDFs from Supabase Storage.

Sources:
  1. deal_documents table → secure-pdfs bucket (deal-level uploads)
  2. applications.payload.uploaded_statements → application-docs bucket (lead-level uploads)

Usage:
    python export_pdfs.py                    # download all
    python export_pdfs.py --limit 100        # first 100
    python export_pdfs.py --source deals     # only deal documents
    python export_pdfs.py --source apps      # only application uploads
    python export_pdfs.py --local-dir /path  # also copy from local dir
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_KEY")

DATA_DIR = PROJECT_DIR / "data" / "pdfs"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"files": {}, "stats": {"total": 0, "deals": 0, "apps": 0, "skipped": 0}}


def _save_manifest(manifest: dict):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def _safe_filename(name: str) -> str:
    """Sanitize filename for filesystem."""
    return "".join(c if c.isalnum() or c in ".-_" else "_" for c in name)


def export_deal_documents(sb, manifest: dict, limit: int = None) -> int:
    """Download PDFs from deal_documents table → secure-pdfs bucket."""
    print("\n--- Deal Documents (secure-pdfs) ---")

    offset = 0
    page_size = 500
    downloaded = 0
    total_found = 0

    while True:
        query = (
            sb.table("deal_documents")
            .select("id, deal_id, file_name, storage_path, bucket, content_type")
            .order("id")
            .range(offset, offset + page_size - 1)
        )
        result = query.execute()
        rows = result.data or []
        if not rows:
            break

        for row in rows:
            content_type = row.get("content_type", "")
            file_name = row.get("file_name", "")

            # Only PDFs
            if "pdf" not in content_type.lower() and not file_name.lower().endswith(".pdf"):
                continue

            total_found += 1
            storage_path = row.get("storage_path", "")
            bucket = row.get("bucket", "secure-pdfs")
            doc_id = row["id"]

            # Skip if already downloaded
            manifest_key = f"deal_{doc_id}"
            if manifest_key in manifest["files"]:
                continue

            if limit and downloaded >= limit:
                break

            safe_name = _safe_filename(f"deal_{row['deal_id']}_{file_name}")
            local_path = DATA_DIR / safe_name

            try:
                data = sb.storage.from_(bucket).download(storage_path)
                local_path.write_bytes(data)
                manifest["files"][manifest_key] = {
                    "filename": safe_name,
                    "source": "deal_documents",
                    "deal_id": row["deal_id"],
                    "storage_path": storage_path,
                    "bucket": bucket,
                    "original_name": file_name,
                }
                downloaded += 1
                manifest["stats"]["deals"] += 1
                if downloaded % 50 == 0:
                    print(f"  Downloaded {downloaded} deal PDFs...")
            except Exception as e:
                print(f"  Error downloading deal doc {doc_id}: {e}")
                manifest["stats"]["skipped"] += 1

        offset += page_size
        if limit and downloaded >= limit:
            break

    print(f"  Found {total_found} deal PDFs, downloaded {downloaded} new")
    return downloaded


def export_application_statements(sb, manifest: dict, limit: int = None) -> int:
    """Download PDFs from applications.payload.uploaded_statements → application-docs bucket."""
    print("\n--- Application Statements (application-docs) ---")

    offset = 0
    page_size = 500
    downloaded = 0
    total_found = 0

    while True:
        result = (
            sb.table("applications")
            .select("id, payload")
            .order("id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = result.data or []
        if not rows:
            break

        for row in rows:
            payload = row.get("payload") or {}
            statements = payload.get("uploaded_statements") or []
            if not statements:
                continue

            app_id = row["id"]

            for idx, stmt in enumerate(statements):
                file_name = stmt.get("filename", "")
                storage_path = stmt.get("storage_path", "")

                if not storage_path or not file_name.lower().endswith(".pdf"):
                    continue

                total_found += 1
                manifest_key = f"app_{app_id}_{idx}"

                if manifest_key in manifest["files"]:
                    continue

                if limit and downloaded >= limit:
                    break

                safe_name = _safe_filename(f"app_{app_id}_{file_name}")
                local_path = DATA_DIR / safe_name

                # Try application-docs first, then secure-pdfs
                data = None
                for bucket in ["application-docs", "secure-pdfs"]:
                    try:
                        data = sb.storage.from_(bucket).download(storage_path)
                        break
                    except Exception:
                        continue

                if data:
                    local_path.write_bytes(data)
                    manifest["files"][manifest_key] = {
                        "filename": safe_name,
                        "source": "applications",
                        "application_id": app_id,
                        "storage_path": storage_path,
                        "bucket": bucket,
                        "original_name": file_name,
                    }
                    downloaded += 1
                    manifest["stats"]["apps"] += 1
                    if downloaded % 50 == 0:
                        print(f"  Downloaded {downloaded} application PDFs...")
                else:
                    print(f"  Could not download app {app_id} stmt: {storage_path}")
                    manifest["stats"]["skipped"] += 1

            if limit and downloaded >= limit:
                break

        offset += page_size
        if limit and downloaded >= limit:
            break

    print(f"  Found {total_found} application PDFs, downloaded {downloaded} new")
    return downloaded


def copy_local_pdfs(local_dir: str, manifest: dict, limit: int = None) -> int:
    """Copy PDFs from a local directory."""
    print(f"\n--- Local PDFs ({local_dir}) ---")
    import shutil

    source = Path(local_dir)
    if not source.is_dir():
        print(f"  Directory not found: {local_dir}")
        return 0

    downloaded = 0
    for pdf_path in sorted(source.glob("**/*.pdf")):
        manifest_key = f"local_{pdf_path.name}"
        if manifest_key in manifest["files"]:
            continue

        if limit and downloaded >= limit:
            break

        safe_name = _safe_filename(f"local_{pdf_path.name}")
        dest = DATA_DIR / safe_name
        shutil.copy2(pdf_path, dest)

        manifest["files"][manifest_key] = {
            "filename": safe_name,
            "source": "local",
            "original_path": str(pdf_path),
            "original_name": pdf_path.name,
        }
        downloaded += 1

    print(f"  Copied {downloaded} local PDFs")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Export bank statement PDFs from Supabase")
    parser.add_argument("--limit", type=int, help="Max PDFs to download per source")
    parser.add_argument("--source", choices=["deals", "apps", "local", "all"], default="all")
    parser.add_argument("--local-dir", help="Local directory with additional PDFs")
    args = parser.parse_args()

    if args.source != "local" and (not SUPABASE_URL or not SUPABASE_KEY):
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE in .env")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()

    total = 0

    if args.source in ("deals", "all"):
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        total += export_deal_documents(sb, manifest, limit=args.limit)

    if args.source in ("apps", "all"):
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        total += export_application_statements(sb, manifest, limit=args.limit)

    if args.local_dir or args.source == "local":
        if args.local_dir:
            total += copy_local_pdfs(args.local_dir, manifest, limit=args.limit)
        else:
            print("Use --local-dir to specify a local PDF directory")

    manifest["stats"]["total"] = len(manifest["files"])
    _save_manifest(manifest)

    print(f"\nDone. Total PDFs in data/pdfs/: {manifest['stats']['total']}")
    print(f"  From deals: {manifest['stats']['deals']}")
    print(f"  From apps:  {manifest['stats']['apps']}")
    print(f"  Skipped:    {manifest['stats']['skipped']}")


if __name__ == "__main__":
    main()
