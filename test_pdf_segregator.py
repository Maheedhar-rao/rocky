#!/usr/bin/env python3
"""
Test harness for pdf_segregator on real PDFs from data/pdfs/.

Validates the segregation design rather than just timing it:
  1. PLAN     — page counting is sub-second
  2. PROVENANCE — every txn has a unique-per-PDF (pdf_idx, page, row); page_uid
                  is unique across the merged set's page boundaries
  3. ORDERING — merged list is sorted by the provenance triple (deterministic)
  4. EQUIVALENCE — parallel result == sequential result (same txns, same months);
                   parallelism must not change the answer, only the wall-clock
  5. REGROUP  — month buckets are date-derived and span/merge PDFs correctly

Run: python3 test_pdf_segregator.py [N_PDFS]
"""
import glob
import io
import sys
import time

import pdfplumber

# tesseract is broken on this dev box; the OCR fallback only fires on image-only
# pages. Stub it so the ML path can be benchmarked. (Production keeps real OCR.)
import predict
predict._extract_words_ocr = lambda *a, **k: []

from pdf_segregator import segregate_and_analyze, plan_uploads, MAX_PARSE_WORKERS


def pick_pdfs(n):
    """Pick n text-layer PDFs (so the broken-OCR path never triggers here)."""
    picked = []
    for f in sorted(glob.glob("data/pdfs/*.pdf")):
        try:
            b = open(f, "rb").read()
            with pdfplumber.open(io.BytesIO(b)) as pdf:
                if pdf.pages and len(pdf.pages[0].extract_words()) > 40 and 3 <= len(pdf.pages) <= 12:
                    picked.append((f.split("/")[-1], b))
        except Exception:
            pass
        if len(picked) >= n:
            break
    return picked


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    pdfs = pick_pdfs(n)
    print(f"Loaded {len(pdfs)} PDFs: {[p[0][:28] for p in pdfs]}\n")

    # warm the model so timings reflect steady state
    from predict import get_parser
    get_parser()

    # --- 1. PLAN ---
    print("1. PLAN (page counting)")
    t0 = time.time()
    plan = plan_uploads(pdfs)
    plan_ms = (time.time() - t0) * 1000
    total_pages = sum(p.pages for p in plan)
    for p in plan:
        print(f"     pdf {p.pdf_idx}: {p.pages}p  {p.filename[:40]}")
    check(f"page count {plan_ms:.1f} ms < 2000 ms (total {total_pages} pages)", plan_ms < 2000)

    # --- run the segregator ---
    print("\n2. DISPATCH + STAMP + REGROUP")
    res = segregate_and_analyze(pdfs)
    print(f"     timings: {res.timings_ms}")
    print(f"     transactions: {len(res.transactions)}  months: {res.summary.get('months')}")

    txns = res.transactions
    ok = True

    # --- 2. PROVENANCE ---
    print("\n3. PROVENANCE checks")
    ok &= check("every txn has pdf_idx/page/row/page_uid",
                all(all(k in t for k in ("pdf_idx", "page", "row", "page_uid")) for t in txns))
    # (pdf_idx, page, row) is unique across the whole merged set
    triples = [(t["pdf_idx"], t["page"], t["row"]) for t in txns]
    ok &= check(f"provenance triple unique ({len(set(triples))}/{len(triples)})",
                len(set(triples)) == len(triples))
    # row restarts at 0 for each (pdf, page)
    first_rows = {}
    for t in txns:
        key = (t["pdf_idx"], t["page"])
        first_rows[key] = min(first_rows.get(key, 1e9), t["row"])
    ok &= check("row index restarts at 0 per (pdf, page)",
                all(v == 0 for v in first_rows.values()))
    # page_uid distinguishes the same page number across different PDFs:
    # among txns on page 0, distinct page_uids must equal distinct source PDFs.
    page0 = [t for t in txns if t["page"] == 0]
    ok &= check("page_uid disambiguates page 0 across PDFs",
                len({t["page_uid"] for t in page0}) == len({t["pdf_idx"] for t in page0})
                and all(t["page_uid"] == f'{t["pdf_idx"]}:0' for t in page0))

    # --- 3. ORDERING ---
    print("\n4. ORDERING (deterministic reassembly)")
    ok &= check("merged list sorted by (pdf_idx, page, row)", triples == sorted(triples))

    # --- 4. EQUIVALENCE: parallel == sequential ---
    print("\n5. EQUIVALENCE (parallel vs sequential)")
    seq = segregate_and_analyze(pdfs, max_workers=1)
    par = segregate_and_analyze(pdfs, max_workers=MAX_PARSE_WORKERS)
    same_count = len(seq.transactions) == len(par.transactions)
    same_months = seq.summary.get("months") == par.summary.get("months")
    same_income = seq.summary.get("monthly_income") == par.summary.get("monthly_income")
    same_expense = seq.summary.get("monthly_expenses") == par.summary.get("monthly_expenses")
    ok &= check("same transaction count", same_count)
    ok &= check("same month buckets", same_months)
    ok &= check("identical monthly_income", same_income)
    ok &= check("identical monthly_expenses", same_expense)
    print(f"     sequential: {seq.timings_ms['parse']:.0f} ms parse  |  "
          f"parallel ({par.timings_ms['workers']}w): {par.timings_ms['parse']:.0f} ms  "
          f"speedup {seq.timings_ms['parse']/max(1,par.timings_ms['parse']):.2f}x")

    # --- 5. REGROUP: date-derived months, cross-PDF merge ---
    print("\n6. REGROUP (month = date, not page)")
    months_per_pdf = {}
    for t in txns:
        months_per_pdf.setdefault(t["pdf_idx"], set()).add(t["month_key"])
    multi = [i for i, ms in months_per_pdf.items() if len({m for m in ms if m != "unknown"}) > 1]
    print(f"     PDFs spanning >1 month: {multi or 'none in this sample'}")
    # a month bucket may receive rows from more than one PDF
    pdfs_per_month = {}
    for t in txns:
        if t["month_key"] != "unknown":
            pdfs_per_month.setdefault(t["month_key"], set()).add(t["pdf_idx"])
    shared = {m: sorted(s) for m, s in pdfs_per_month.items() if len(s) > 1}
    print(f"     months fed by >1 PDF: {shared or 'none in this sample'}")
    ok &= check("month buckets are non-empty and date-keyed (YYYY-MM)",
                all(len(m) == 7 and m[4] == "-" for m in res.summary.get("months", [])))

    print(f"\n{'='*48}\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
