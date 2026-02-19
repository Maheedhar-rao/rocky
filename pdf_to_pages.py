#!/usr/bin/env python3
"""
Convert bank statement PDFs to page images + pdfplumber word bounding boxes.

For each PDF, produces per-page:
  - page_{idx}.png          (200 DPI image for Claude Vision + LayoutLMv3)
  - page_{idx}_words.json   (word tokens with normalized bounding boxes)

Bounding boxes are normalized to 0-1000 scale for LayoutLMv3 input.
If pdfplumber extracts <20 words, falls back to pytesseract OCR.

Usage:
    python pdf_to_pages.py                    # process all PDFs in data/pdfs/
    python pdf_to_pages.py --limit 10         # first 10 PDFs
    python pdf_to_pages.py --pdf path/to.pdf  # single PDF
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
PDF_DIR = PROJECT_DIR / "data" / "pdfs"
PAGES_DIR = PROJECT_DIR / "data" / "pages"

DPI = 200
MIN_WORDS_THRESHOLD = 20  # below this, use OCR fallback


def normalize_bbox(bbox: tuple, page_width: float, page_height: float) -> list:
    """Normalize bounding box to 0-1000 scale for LayoutLMv3."""
    x0, y0, x1, y1 = bbox
    return [
        int(max(0, min(1000, x0 / page_width * 1000))),
        int(max(0, min(1000, y0 / page_height * 1000))),
        int(max(0, min(1000, x1 / page_width * 1000))),
        int(max(0, min(1000, y1 / page_height * 1000))),
    ]


def extract_words_pdfplumber(page) -> list:
    """Extract words with bounding boxes using pdfplumber."""
    words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
    )

    page_width = page.width
    page_height = page.height

    result = []
    for w in words:
        text = w["text"].strip()
        if not text:
            continue
        bbox = (w["x0"], w["top"], w["x1"], w["bottom"])
        norm_bbox = normalize_bbox(bbox, page_width, page_height)
        result.append({
            "text": text,
            "bbox": norm_bbox,
            "top": round(w["top"], 2),
            "bottom": round(w["bottom"], 2),
        })

    return result


def extract_words_ocr(image: Image.Image) -> list:
    """Fallback: extract words using pytesseract OCR."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    img_width, img_height = image.size

    result = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        if not text or int(data["conf"][i]) < 30:
            continue

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        bbox = normalize_bbox((x, y, x + w, y + h), img_width, img_height)

        result.append({
            "text": text,
            "bbox": bbox,
            "top": round(y / img_height * 1000, 2),
            "bottom": round((y + h) / img_height * 1000, 2),
        })

    return result


def process_pdf(pdf_path: Path, output_dir: Path = None) -> dict:
    """Process a single PDF into page images + word JSONs."""
    stem = pdf_path.stem
    if output_dir is None:
        output_dir = PAGES_DIR / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"pages": 0, "words_total": 0, "ocr_pages": 0}

    # Convert PDF to images
    try:
        images = convert_from_path(str(pdf_path), dpi=DPI)
    except Exception as e:
        print(f"  Error converting {pdf_path.name}: {e}")
        return stats

    # Extract words from each page
    try:
        pdf = pdfplumber.open(str(pdf_path))
    except Exception as e:
        print(f"  Error opening {pdf_path.name} with pdfplumber: {e}")
        pdf = None

    for idx, image in enumerate(images):
        # Save page image
        img_path = output_dir / f"page_{idx}.png"
        image.save(str(img_path), "PNG")

        # Extract words
        words = []
        used_ocr = False

        if pdf and idx < len(pdf.pages):
            words = extract_words_pdfplumber(pdf.pages[idx])

        if len(words) < MIN_WORDS_THRESHOLD:
            ocr_words = extract_words_ocr(image)
            if len(ocr_words) > len(words):
                words = ocr_words
                used_ocr = True

        if used_ocr:
            stats["ocr_pages"] += 1

        # Save word JSON
        words_path = output_dir / f"page_{idx}_words.json"
        word_data = {
            "pdf": pdf_path.name,
            "page_index": idx,
            "total_pages": len(images),
            "used_ocr": used_ocr,
            "word_count": len(words),
            "words": words,
        }
        with open(words_path, "w") as f:
            json.dump(word_data, f, indent=2)

        stats["pages"] += 1
        stats["words_total"] += len(words)

    if pdf:
        pdf.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to page images + word bounding boxes")
    parser.add_argument("--limit", type=int, help="Max PDFs to process")
    parser.add_argument("--pdf", help="Process a single PDF file")
    args = parser.parse_args()

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"File not found: {args.pdf}")
            sys.exit(1)
        print(f"Processing {pdf_path.name}...")
        stats = process_pdf(pdf_path)
        print(f"  Pages: {stats['pages']}, Words: {stats['words_total']}, OCR pages: {stats['ocr_pages']}")
        return

    if not PDF_DIR.exists():
        print(f"No PDFs directory at {PDF_DIR}. Run export_pdfs.py first.")
        sys.exit(1)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        sys.exit(1)

    if args.limit:
        pdfs = pdfs[:args.limit]

    print(f"Processing {len(pdfs)} PDFs...")
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_words = 0
    total_ocr = 0

    for i, pdf_path in enumerate(pdfs):
        # Skip if already processed
        output_dir = PAGES_DIR / pdf_path.stem
        if output_dir.exists() and any(output_dir.glob("page_*_words.json")):
            existing = list(output_dir.glob("page_*_words.json"))
            total_pages += len(existing)
            continue

        stats = process_pdf(pdf_path)
        total_pages += stats["pages"]
        total_words += stats["words_total"]
        total_ocr += stats["ocr_pages"]

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(pdfs)} PDFs ({total_pages} pages)...")

    print(f"\nDone. {len(pdfs)} PDFs → {total_pages} pages")
    print(f"  Total words extracted: {total_words}")
    print(f"  OCR fallback pages:   {total_ocr}")


if __name__ == "__main__":
    main()
