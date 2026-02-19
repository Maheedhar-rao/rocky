#!/usr/bin/env python3
"""
Detect bank names across a sample of PDFs by:
1. Checking filename patterns
2. Extracting first-page text with pdfplumber and searching for bank names
"""

import os
import re
import random
import pdfplumber
from collections import defaultdict, Counter
import time

PDF_DIR = "/Users/maheedharraogovada/Desktop/ml-statement-parser/data/pdfs/"

# Bank detection patterns: (display_name, [filename_patterns], [text_patterns])
# Text patterns are searched case-insensitively in the first page text
BANKS = [
    ("Chase / JPMorgan", 
     [r"(?i)chase", r"(?i)jpmorgan"],
     [r"(?i)jpmorgan\s*chase", r"(?i)chase\b.*bank", r"(?i)\bchase\b"]),
    
    ("Wells Fargo",
     [r"(?i)wells\s*fargo", r"(?i)WF[\s_]"],
     [r"(?i)wells\s*fargo"]),
    
    ("Bank of America",
     [r"(?i)bank\s*of\s*america", r"(?i)\bBofA\b", r"(?i)\bBOA\b"],
     [r"(?i)bank\s*of\s*america"]),
    
    ("Huntington",
     [r"(?i)huntington"],
     [r"(?i)huntington\s*(national\s*)?bank", r"(?i)huntington\b"]),
    
    ("US Bank (USB)",
     [r"(?i)\bUSB\b", r"(?i)us\s*bank"],
     [r"(?i)u\.?s\.?\s*bank", r"(?i)us\s*bancorp"]),
    
    ("Citibank / Citi",
     [r"(?i)citi\s*bank", r"(?i)\bciti\b"],
     [r"(?i)citi\s*bank", r"(?i)citigroup", r"(?i)\bciti\b"]),
    
    ("TD Bank",
     [r"(?i)\bTD[\s_]bank", r"(?i)\bTD[\s_]"],
     [r"(?i)td\s*bank", r"(?i)td\s*ameritrade"]),
    
    ("PNC",
     [r"(?i)\bPNC\b"],
     [r"(?i)\bpnc\s*(bank|financial)?"]),
    
    ("Capital One",
     [r"(?i)capital\s*one"],
     [r"(?i)capital\s*one"]),
    
    ("Truist",
     [r"(?i)truist"],
     [r"(?i)truist"]),
    
    ("Regions",
     [r"(?i)regions\s*(bank|financial)?"],
     [r"(?i)regions\s*(bank|financial)"]),
    
    ("Fifth Third",
     [r"(?i)fifth\s*third"],
     [r"(?i)fifth\s*third"]),
    
    ("KeyBank",
     [r"(?i)key\s*bank"],
     [r"(?i)key\s*bank", r"(?i)keycorp"]),
    
    ("M&T Bank",
     [r"(?i)m\s*[&]\s*t\s*bank"],
     [r"(?i)m\s*[&]\s*t\s*bank"]),
    
    ("Citizens",
     [r"(?i)citizens\s*(bank)?"],
     [r"(?i)citizens\s*(bank|financial)"]),
    
    ("BMO / Harris",
     [r"(?i)\bBMO\b", r"(?i)harris\s*bank"],
     [r"(?i)\bbmo\b", r"(?i)harris\s*bank"]),
    
    ("Ally Bank",
     [r"(?i)ally\s*bank", r"(?i)\bally\b"],
     [r"(?i)ally\s*(bank|financial)"]),
    
    ("Discover Bank",
     [r"(?i)discover"],
     [r"(?i)discover\s*bank"]),
    
    ("USAA",
     [r"(?i)\bUSAA\b"],
     [r"(?i)\busaa\b"]),
    
    ("Navy Federal",
     [r"(?i)navy\s*federal"],
     [r"(?i)navy\s*federal"]),
    
    ("Amegy",
     [r"(?i)amegy"],
     [r"(?i)amegy"]),
    
    ("First National",
     [r"(?i)first\s*national"],
     [r"(?i)first\s*national"]),

    ("Zions / NBAZ",
     [r"(?i)zions", r"(?i)\bnbaz\b"],
     [r"(?i)zions\s*(bank|bancorp)?", r"(?i)national\s*bank\s*of\s*arizona"]),

    ("Comerica",
     [r"(?i)comerica"],
     [r"(?i)comerica"]),

    ("First Horizon",
     [r"(?i)first\s*horizon"],
     [r"(?i)first\s*horizon"]),

    ("Webster Bank",
     [r"(?i)webster"],
     [r"(?i)webster\s*bank"]),

    ("Frost Bank",
     [r"(?i)frost"],
     [r"(?i)frost\s*bank"]),

    ("Federal Credit Union (generic)",
     [r"(?i)credit\s*union"],
     [r"(?i)federal\s*credit\s*union", r"(?i)\bcredit\s*union\b"]),

    ("Community / Local Bank (generic)",
     [],
     [r"(?i)community\s*bank", r"(?i)savings\s*bank"]),
]


def detect_bank_filename(filename):
    """Try to detect bank from filename alone."""
    matches = []
    for bank_name, fn_patterns, _ in BANKS:
        for pat in fn_patterns:
            if re.search(pat, filename):
                matches.append(bank_name)
                break
    return matches


def detect_bank_text(text):
    """Detect bank from first-page text."""
    matches = []
    for bank_name, _, text_patterns in BANKS:
        for pat in text_patterns:
            if re.search(pat, text):
                matches.append(bank_name)
                break
    return matches


def main():
    all_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    total = len(all_files)
    print(f"Total PDFs found: {total}")
    
    # --- Phase 1: Filename-based detection on ALL files ---
    print("\n" + "="*70)
    print("PHASE 1: Filename-based detection (all files)")
    print("="*70)
    
    fn_bank_files = defaultdict(list)  # bank -> [filenames]
    fn_unmatched = []
    
    for f in all_files:
        banks = detect_bank_filename(f)
        if banks:
            for b in banks:
                fn_bank_files[b].append(f)
        else:
            fn_unmatched.append(f)
    
    print(f"\nMatched by filename: {total - len(fn_unmatched)} / {total}")
    print(f"Unmatched by filename: {len(fn_unmatched)}")
    
    for bank, files in sorted(fn_bank_files.items(), key=lambda x: -len(x[1])):
        print(f"  {bank:40s} {len(files):5d} files  (e.g. {files[0][:70]})")
    
    # --- Phase 2: Text-based detection on a sample ---
    SAMPLE_SIZE = 200
    sample = random.sample(all_files, min(SAMPLE_SIZE, total))
    
    print(f"\n{'='*70}")
    print(f"PHASE 2: Text-based detection (sample of {len(sample)} PDFs)")
    print("="*70)
    
    text_bank_files = defaultdict(list)
    text_unmatched = []
    errors = []
    
    t0 = time.time()
    for i, fname in enumerate(sample):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ...processed {i+1}/{len(sample)} ({elapsed:.1f}s)")
        
        fpath = os.path.join(PDF_DIR, fname)
        try:
            with pdfplumber.open(fpath) as pdf:
                if len(pdf.pages) == 0:
                    errors.append((fname, "no pages"))
                    continue
                page_text = pdf.pages[0].extract_text() or ""
        except Exception as e:
            errors.append((fname, str(e)[:80]))
            continue
        
        banks = detect_bank_text(page_text)
        if banks:
            for b in banks:
                text_bank_files[b].append(fname)
        else:
            text_unmatched.append((fname, page_text[:200]))
    
    elapsed = time.time() - t0
    print(f"\nText extraction completed in {elapsed:.1f}s")
    print(f"Matched by text: {len(sample) - len(text_unmatched) - len(errors)} / {len(sample)}")
    print(f"Unmatched by text: {len(text_unmatched)}")
    print(f"Errors: {len(errors)}")
    
    print(f"\n--- Bank counts from TEXT detection (sample of {len(sample)}) ---")
    for bank, files in sorted(text_bank_files.items(), key=lambda x: -len(x[1])):
        pct = len(files) / len(sample) * 100
        est_total = int(len(files) / len(sample) * total)
        print(f"  {bank:40s} {len(files):4d} in sample ({pct:5.1f}%)  ~{est_total:5d} estimated total  (e.g. {files[0][:60]})")
    
    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for fname, err in errors[:10]:
            print(f"  {fname[:60]:60s} -> {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    if text_unmatched:
        print(f"\n--- Unmatched by text ({len(text_unmatched)}) - first-page snippet ---")
        for fname, snippet in text_unmatched[:15]:
            clean = snippet.replace("\n", " | ")[:120]
            print(f"  {fname[:55]:55s} -> {clean}")
        if len(text_unmatched) > 15:
            print(f"  ... and {len(text_unmatched) - 15} more")
    
    # --- Phase 3: Combined summary ---
    print(f"\n{'='*70}")
    print("PHASE 3: Combined summary (filename on all + text on sample)")
    print("="*70)
    
    # Merge: use filename counts for all, supplement with text-only detections
    all_banks = set(list(fn_bank_files.keys()) + list(text_bank_files.keys()))
    
    print(f"\n{'Bank':<40s} {'Filename (all)':<16s} {'Text (sample)':<16s} {'Est. Total':<12s}")
    print("-" * 84)
    for bank in sorted(all_banks, key=lambda b: -(len(fn_bank_files.get(b, [])) + int(len(text_bank_files.get(b, [])) / len(sample) * total))):
        fn_count = len(fn_bank_files.get(bank, []))
        txt_count = len(text_bank_files.get(bank, []))
        est = max(fn_count, int(txt_count / len(sample) * total))
        print(f"  {bank:<38s} {fn_count:>6d}         {txt_count:>6d}         ~{est:>5d}")


if __name__ == "__main__":
    random.seed(42)
    main()
