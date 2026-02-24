#!/usr/bin/env python3
"""
Model-assisted labeling UI for bank statement pages.

Shows page image + model-predicted transactions. User corrects and saves.
Saved labels are identical to Claude Vision format → align_labels.py works unchanged.

Usage:
    python label_ui.py                    # start on localhost:8501
    python label_ui.py --port 8502        # custom port
    python label_ui.py --no-model         # skip model pre-fill (empty forms)
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

PROJECT_DIR = Path(__file__).resolve().parent
PAGES_DIR = PROJECT_DIR / "data" / "pages"
LABELS_DIR = PROJECT_DIR / "data" / "labels"

# Lazy-loaded model
_parser = None
_use_model = True


def _get_parser():
    global _parser
    if _parser is None and _use_model:
        try:
            from predict import get_parser
            _parser = get_parser()
        except Exception as e:
            print(f"Warning: could not load model: {e}")
    return _parser


def _detect_bank(pdf_stem: str) -> str:
    """Detect bank from first page words."""
    words_path = PAGES_DIR / pdf_stem / "page_0_words.json"
    if not words_path.exists():
        return "Unknown"
    try:
        with open(words_path) as f:
            data = json.load(f)
        text = " ".join(w["text"].lower() for w in data.get("words", [])[:100])
        from predict import _BANK_PATTERNS
        for pattern, bank in _BANK_PATTERNS.items():
            if pattern in text:
                return bank
        return "Unknown"
    except Exception:
        return "Unknown"


def _get_page_count(pdf_stem: str) -> int:
    """Count pages for a PDF stem."""
    pdf_dir = PAGES_DIR / pdf_stem
    if not pdf_dir.exists():
        return 0
    return len(list(pdf_dir.glob("page_*.png")))


def _predict_page_transactions(pdf_stem: str, page_idx: int) -> list:
    """Run model inference on a single page, return transaction dicts."""
    parser = _get_parser()
    if parser is None:
        return []

    words_path = PAGES_DIR / pdf_stem / f"page_{page_idx}_words.json"
    image_path = PAGES_DIR / pdf_stem / f"page_{page_idx}.png"

    if not words_path.exists() or not image_path.exists():
        return []

    try:
        from PIL import Image
        from predict import _bio_to_transactions, _classify_credit_debit

        with open(words_path) as f:
            words_data = json.load(f)

        words = words_data.get("words", [])
        if not words:
            return []

        image = Image.open(image_path).convert("RGB")
        tags, confidences = parser._predict_page(words, words, image)

        word_texts = [w["text"] for w in words[:len(tags)]]
        transactions = _bio_to_transactions(word_texts, tags, page_idx, confidences)
        _classify_credit_debit(transactions)

        return [
            {
                "date": t.date,
                "description": t.description,
                "amount": t.amount,
                "type": t.type,
                "balance_after": t.balance_after,
            }
            for t in transactions
        ]
    except Exception as e:
        print(f"Prediction error for {pdf_stem}/page_{page_idx}: {e}")
        return []


app = FastAPI(title="Statement Labeling UI")


@app.get("/api/stats")
def api_stats():
    """Return labeling progress stats."""
    labeled_stems = set()
    if LABELS_DIR.exists():
        labeled_stems = {d.name for d in LABELS_DIR.iterdir() if d.is_dir()}

    all_stems = set()
    if PAGES_DIR.exists():
        all_stems = {d.name for d in PAGES_DIR.iterdir() if d.is_dir()}

    # Count pages
    labeled_pages = 0
    for stem in labeled_stems:
        label_dir = LABELS_DIR / stem
        labeled_pages += len(list(label_dir.glob("page_*_labels.json")))

    total_pages = 0
    for stem in all_stems:
        total_pages += len(list((PAGES_DIR / stem).glob("page_*.png")))

    return {
        "labeled_pdfs": len(labeled_stems),
        "total_pdfs": len(all_stems),
        "labeled_pages": labeled_pages,
        "total_pages": total_pages,
    }


@app.get("/api/queue")
def api_queue():
    """Return unlabeled PDFs sorted by bank diversity priority."""
    labeled_stems = set()
    if LABELS_DIR.exists():
        labeled_stems = {d.name for d in LABELS_DIR.iterdir() if d.is_dir()}

    # Get all PDF stems with pages
    all_stems = []
    if PAGES_DIR.exists():
        for d in sorted(PAGES_DIR.iterdir()):
            if d.is_dir() and d.name not in labeled_stems:
                page_count = len(list(d.glob("page_*.png")))
                if page_count > 0:
                    all_stems.append({"stem": d.name, "pages": page_count})

    # Detect banks (sample first 200 for speed)
    bank_counts = {}
    queue = []
    for item in all_stems[:500]:
        bank = _detect_bank(item["stem"])
        item["bank"] = bank
        bank_counts[bank] = bank_counts.get(bank, 0) + 1
        queue.append(item)

    # Sort: underrepresented banks first
    queue.sort(key=lambda x: (bank_counts.get(x["bank"], 0), x["stem"]))

    return {
        "queue": queue[:200],  # cap for UI performance
        "bank_counts": bank_counts,
        "total_unlabeled": len(all_stems),
    }


@app.get("/api/page/{pdf_stem}/{page_idx}")
def api_page(pdf_stem: str, page_idx: int):
    """Return page data + predicted transactions for labeling."""
    image_path = PAGES_DIR / pdf_stem / f"page_{page_idx}.png"
    if not image_path.exists():
        return JSONResponse({"error": "Page not found"}, status_code=404)

    # Check if label already exists
    label_path = LABELS_DIR / pdf_stem / f"page_{page_idx}_labels.json"
    existing_label = None
    if label_path.exists():
        with open(label_path) as f:
            existing_label = json.load(f)

    # Get total pages for this PDF
    total_pages = _get_page_count(pdf_stem)

    # Pre-fill: existing label > model prediction > empty
    if existing_label:
        transactions = existing_label.get("transactions", [])
        source = "existing_label"
        metadata = existing_label.get("metadata", {})
    else:
        transactions = _predict_page_transactions(pdf_stem, page_idx)
        source = "model" if transactions else "empty"
        metadata = {}

    return {
        "pdf_stem": pdf_stem,
        "page_idx": page_idx,
        "total_pages": total_pages,
        "image_url": f"/api/image/{pdf_stem}/{page_idx}",
        "transactions": transactions,
        "source": source,
        "metadata": metadata,
    }


@app.get("/api/image/{pdf_stem}/{page_idx}")
def api_image(pdf_stem: str, page_idx: int):
    """Serve a page image."""
    image_path = PAGES_DIR / pdf_stem / f"page_{page_idx}.png"
    if not image_path.exists():
        return JSONResponse({"error": "Image not found"}, status_code=404)
    return FileResponse(image_path, media_type="image/png")


@app.post("/api/save/{pdf_stem}/{page_idx}")
async def api_save(pdf_stem: str, page_idx: int, body: dict):
    """Save corrected transactions as a label JSON file."""
    transactions = body.get("transactions", [])
    metadata = body.get("metadata", {})

    label_dir = LABELS_DIR / pdf_stem
    label_dir.mkdir(parents=True, exist_ok=True)

    label_data = {
        "metadata": metadata,
        "transactions": transactions,
        "has_transactions": len(transactions) > 0,
        "notes": "manually labeled via label_ui",
        "label_source": "manual",
        "_source_image": f"page_{page_idx}.png",
        "_pdf_stem": pdf_stem,
    }

    label_path = label_dir / f"page_{page_idx}_labels.json"
    with open(label_path, "w") as f:
        json.dump(label_data, f, indent=2)

    # Find next unlabeled page in this PDF
    next_page = None
    total_pages = _get_page_count(pdf_stem)
    for p in range(page_idx + 1, total_pages):
        next_label = label_dir / f"page_{p}_labels.json"
        if not next_label.exists():
            next_page = p
            break

    return {
        "saved": True,
        "path": str(label_path),
        "next_page": next_page,
        "next_pdf_stem": pdf_stem if next_page is not None else None,
    }


LABEL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Statement Labeling UI</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }
.top-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 16px; background: #16213e; border-bottom: 1px solid #0f3460;
}
.top-bar h1 { font-size: 16px; color: #e94560; }
.stats { font-size: 13px; color: #a0a0b0; }
.stats span { color: #e94560; font-weight: bold; }
.main { display: flex; height: calc(100vh - 44px); }
.sidebar {
    width: 260px; background: #16213e; border-right: 1px solid #0f3460;
    overflow-y: auto; flex-shrink: 0;
}
.sidebar h3 { padding: 10px 12px 6px; font-size: 13px; color: #a0a0b0; }
.sidebar input {
    width: calc(100% - 24px); margin: 0 12px 8px; padding: 6px 8px;
    background: #1a1a2e; border: 1px solid #0f3460; color: #e0e0e0;
    border-radius: 4px; font-size: 13px;
}
.pdf-item {
    padding: 6px 12px; cursor: pointer; font-size: 13px;
    border-bottom: 1px solid #1a1a2e; display: flex; justify-content: space-between;
}
.pdf-item:hover { background: #1a1a2e; }
.pdf-item.active { background: #0f3460; color: #e94560; }
.pdf-item .bank { font-size: 11px; color: #a0a0b0; }
.content { flex: 1; display: flex; overflow: hidden; }
.image-panel {
    flex: 1; overflow: auto; padding: 8px; background: #111;
    display: flex; justify-content: center;
}
.image-panel img { max-width: 100%; height: auto; }
.label-panel {
    width: 520px; flex-shrink: 0; display: flex; flex-direction: column;
    border-left: 1px solid #0f3460; background: #16213e;
}
.label-header {
    padding: 10px 12px; border-bottom: 1px solid #0f3460;
    display: flex; justify-content: space-between; align-items: center;
}
.label-header .page-nav button {
    background: #0f3460; border: none; color: #e0e0e0; padding: 4px 10px;
    border-radius: 3px; cursor: pointer; margin-left: 4px; font-size: 13px;
}
.label-header .page-nav button:hover { background: #e94560; }
.label-header .source-badge {
    font-size: 11px; padding: 2px 8px; border-radius: 10px;
    background: #0f3460; color: #a0a0b0;
}
.label-header .source-badge.model { background: #1b4332; color: #95d5b2; }
.label-header .source-badge.existing_label { background: #3a0ca3; color: #b8c0ff; }
.txn-table-wrap { flex: 1; overflow-y: auto; padding: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; padding: 4px 6px; color: #a0a0b0; font-size: 11px; border-bottom: 1px solid #0f3460; position: sticky; top: 0; background: #16213e; }
td { padding: 3px 4px; border-bottom: 1px solid #1a1a2e; }
td input, td select {
    width: 100%; background: #1a1a2e; border: 1px solid #0f3460;
    color: #e0e0e0; padding: 4px 6px; border-radius: 3px; font-size: 12px;
}
td input:focus, td select:focus { border-color: #e94560; outline: none; }
td .del-btn {
    background: none; border: none; color: #e94560; cursor: pointer;
    font-size: 16px; padding: 0 4px;
}
.actions {
    padding: 10px 12px; border-top: 1px solid #0f3460;
    display: flex; gap: 8px; flex-wrap: wrap;
}
.actions button {
    padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer;
    font-size: 13px; font-weight: 600;
}
.btn-save { background: #e94560; color: #fff; }
.btn-save:hover { background: #c73e54; }
.btn-skip { background: #0f3460; color: #e0e0e0; }
.btn-skip:hover { background: #1a4a7a; }
.btn-add { background: #1b4332; color: #95d5b2; }
.btn-add:hover { background: #2d6a4f; }
.btn-no-txn { background: #533a0f; color: #f0c040; }
.btn-no-txn:hover { background: #6b4a12; }
.toast {
    position: fixed; bottom: 20px; right: 20px; padding: 10px 20px;
    background: #1b4332; color: #95d5b2; border-radius: 6px;
    font-size: 14px; display: none; z-index: 999;
}
.loading { text-align: center; padding: 40px; color: #a0a0b0; }
.meta-row { padding: 6px 12px; font-size: 12px; color: #a0a0b0; border-bottom: 1px solid #0f3460; }
.meta-row input {
    background: #1a1a2e; border: 1px solid #0f3460; color: #e0e0e0;
    padding: 3px 6px; border-radius: 3px; font-size: 12px; width: 220px; margin-left: 6px;
}
.keyboard-hint { font-size: 11px; color: #555; padding: 4px 12px; }
</style>
</head>
<body>

<div class="top-bar">
    <h1>Statement Labeling</h1>
    <div class="stats">
        Labeled: <span id="stat-pages">-</span> pages across <span id="stat-pdfs">-</span> PDFs
        | Unlabeled: <span id="stat-unlabeled">-</span> PDFs
    </div>
</div>

<div class="main">
    <div class="sidebar">
        <h3>Unlabeled PDFs</h3>
        <input type="text" id="filter" placeholder="Filter by name or bank..." oninput="filterQueue()">
        <div id="queue-list"></div>
    </div>

    <div class="content">
        <div class="image-panel" id="image-panel">
            <div class="loading">Select a PDF from the sidebar to begin labeling.</div>
        </div>

        <div class="label-panel">
            <div class="label-header">
                <div>
                    <strong id="current-stem">-</strong>
                    <span class="source-badge" id="source-badge">-</span>
                </div>
                <div class="page-nav">
                    <button onclick="prevPage()">&larr;</button>
                    <span id="page-info">-</span>
                    <button onclick="nextPage()">&rarr;</button>
                </div>
            </div>
            <div class="meta-row">
                Bank: <input type="text" id="meta-bank" placeholder="e.g. Chase">
                Period: <input type="text" id="meta-period" placeholder="e.g. Jan 1 - Jan 31, 2024">
            </div>
            <div class="txn-table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th style="width:70px">Date</th>
                            <th>Description</th>
                            <th style="width:80px">Amount</th>
                            <th style="width:70px">Type</th>
                            <th style="width:80px">Balance</th>
                            <th style="width:28px"></th>
                        </tr>
                    </thead>
                    <tbody id="txn-body"></tbody>
                </table>
            </div>
            <div class="keyboard-hint">Ctrl+Enter: Save &amp; Next | Ctrl+Shift+A: Add Row | Ctrl+Shift+S: Skip</div>
            <div class="actions">
                <button class="btn-save" onclick="saveAndNext()">Save &amp; Next</button>
                <button class="btn-add" onclick="addRow()">+ Row</button>
                <button class="btn-no-txn" onclick="saveNoTransactions()">No Transactions</button>
                <button class="btn-skip" onclick="skipPage()">Skip</button>
            </div>
        </div>
    </div>
</div>

<div class="toast" id="toast"></div>

<script>
let queue = [];
let currentStem = null;
let currentPage = 0;
let totalPages = 0;

async function loadStats() {
    const r = await fetch('/api/stats');
    const d = await r.json();
    document.getElementById('stat-pages').textContent = d.labeled_pages;
    document.getElementById('stat-pdfs').textContent = d.labeled_pdfs;
    document.getElementById('stat-unlabeled').textContent = d.total_pdfs - d.labeled_pdfs;
}

async function loadQueue() {
    const r = await fetch('/api/queue');
    const d = await r.json();
    queue = d.queue;
    renderQueue(queue);
}

function renderQueue(items) {
    const el = document.getElementById('queue-list');
    el.innerHTML = items.map((item, i) =>
        `<div class="pdf-item ${item.stem === currentStem ? 'active' : ''}"
              onclick="selectPdf('${item.stem}', 0)"
              title="${item.stem}">
            <div>
                <div>${item.stem.length > 28 ? item.stem.slice(0, 28) + '...' : item.stem}</div>
                <div class="bank">${item.bank} | ${item.pages}p</div>
            </div>
        </div>`
    ).join('');
}

function filterQueue() {
    const q = document.getElementById('filter').value.toLowerCase();
    const filtered = queue.filter(i =>
        i.stem.toLowerCase().includes(q) || i.bank.toLowerCase().includes(q)
    );
    renderQueue(filtered);
}

async function selectPdf(stem, pageIdx) {
    currentStem = stem;
    currentPage = pageIdx;
    await loadPage();
    renderQueue(queue);  // update active state
}

async function loadPage() {
    if (!currentStem) return;

    document.getElementById('image-panel').innerHTML = '<div class="loading">Loading...</div>';
    document.getElementById('current-stem').textContent = currentStem;

    const r = await fetch(`/api/page/${encodeURIComponent(currentStem)}/${currentPage}`);
    const d = await r.json();

    if (d.error) {
        document.getElementById('image-panel').innerHTML = `<div class="loading">${d.error}</div>`;
        return;
    }

    totalPages = d.total_pages;
    document.getElementById('page-info').textContent = `${currentPage + 1} / ${totalPages}`;

    // Source badge
    const badge = document.getElementById('source-badge');
    badge.textContent = d.source;
    badge.className = 'source-badge ' + d.source;

    // Image
    document.getElementById('image-panel').innerHTML =
        `<img src="${d.image_url}" alt="Page ${currentPage}">`;

    // Metadata
    document.getElementById('meta-bank').value = d.metadata.bank_name || '';
    document.getElementById('meta-period').value = d.metadata.statement_period || '';

    // Transactions
    const tbody = document.getElementById('txn-body');
    tbody.innerHTML = '';
    if (d.transactions.length === 0) {
        addRow();
    } else {
        d.transactions.forEach(t => addRow(t));
    }
}

function addRow(t) {
    const tbody = document.getElementById('txn-body');
    const tr = document.createElement('tr');
    t = t || { date: '', description: '', amount: '', type: 'debit', balance_after: '' };
    const amt = t.amount !== null && t.amount !== '' && t.amount !== 0 ? t.amount : '';
    const bal = t.balance_after !== null && t.balance_after !== '' ? t.balance_after : '';
    tr.innerHTML = `
        <td><input type="text" class="txn-date" value="${t.date || ''}"></td>
        <td><input type="text" class="txn-desc" value="${(t.description || '').replace(/"/g, '&quot;')}"></td>
        <td><input type="number" step="0.01" class="txn-amount" value="${amt}"></td>
        <td><select class="txn-type">
            <option value="debit" ${t.type === 'debit' ? 'selected' : ''}>debit</option>
            <option value="credit" ${t.type === 'credit' ? 'selected' : ''}>credit</option>
        </select></td>
        <td><input type="number" step="0.01" class="txn-balance" value="${bal}"></td>
        <td><button class="del-btn" onclick="this.closest('tr').remove()">&times;</button></td>
    `;
    tbody.appendChild(tr);
    // Focus the date field of the new row
    tr.querySelector('.txn-date').focus();
}

function collectTransactions() {
    const rows = document.querySelectorAll('#txn-body tr');
    const txns = [];
    rows.forEach(row => {
        const date = row.querySelector('.txn-date').value.trim();
        const desc = row.querySelector('.txn-desc').value.trim();
        const amountStr = row.querySelector('.txn-amount').value.trim();
        const type = row.querySelector('.txn-type').value;
        const balStr = row.querySelector('.txn-balance').value.trim();

        // Skip completely empty rows
        if (!date && !desc && !amountStr) return;

        txns.push({
            date: date,
            description: desc,
            amount: amountStr ? parseFloat(amountStr) : 0,
            type: type,
            balance_after: balStr ? parseFloat(balStr) : null,
        });
    });
    return txns;
}

function collectMetadata() {
    return {
        bank_name: document.getElementById('meta-bank').value.trim(),
        statement_period: document.getElementById('meta-period').value.trim(),
        account_number_last4: null,
        page_number: currentPage + 1,
    };
}

async function saveAndNext() {
    if (!currentStem) return;

    const txns = collectTransactions();
    const metadata = collectMetadata();

    const r = await fetch(`/api/save/${encodeURIComponent(currentStem)}/${currentPage}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transactions: txns, metadata: metadata }),
    });
    const d = await r.json();

    if (d.saved) {
        showToast(`Saved ${txns.length} transactions`);
        loadStats();

        // Go to next unlabeled page or next PDF
        if (d.next_page !== null && d.next_page !== undefined) {
            currentPage = d.next_page;
            await loadPage();
        } else {
            // Move to next PDF in queue
            const idx = queue.findIndex(q => q.stem === currentStem);
            if (idx >= 0 && idx < queue.length - 1) {
                queue.splice(idx, 1);  // remove labeled PDF from queue
                const next = queue[idx] || queue[0];
                if (next) {
                    await selectPdf(next.stem, 0);
                } else {
                    showToast('All done!');
                }
            }
        }
    }
}

async function saveNoTransactions() {
    if (!currentStem) return;

    const metadata = collectMetadata();
    const r = await fetch(`/api/save/${encodeURIComponent(currentStem)}/${currentPage}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transactions: [], metadata: metadata }),
    });
    const d = await r.json();

    if (d.saved) {
        showToast('Saved (no transactions)');
        loadStats();

        if (d.next_page !== null && d.next_page !== undefined) {
            currentPage = d.next_page;
            await loadPage();
        } else {
            const idx = queue.findIndex(q => q.stem === currentStem);
            if (idx >= 0 && idx < queue.length - 1) {
                queue.splice(idx, 1);
                const next = queue[idx] || queue[0];
                if (next) await selectPdf(next.stem, 0);
            }
        }
    }
}

function skipPage() {
    if (currentPage < totalPages - 1) {
        currentPage++;
        loadPage();
    } else {
        // Skip to next PDF
        const idx = queue.findIndex(q => q.stem === currentStem);
        if (idx >= 0 && idx < queue.length - 1) {
            selectPdf(queue[idx + 1].stem, 0);
        }
    }
}

function prevPage() {
    if (currentPage > 0) {
        currentPage--;
        loadPage();
    }
}

function nextPage() {
    if (currentPage < totalPages - 1) {
        currentPage++;
        loadPage();
    }
}

function showToast(msg) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.style.display = 'block';
    setTimeout(() => { el.style.display = 'none'; }, 2000);
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        saveAndNext();
    } else if (e.ctrlKey && e.shiftKey && e.key === 'A') {
        e.preventDefault();
        addRow();
    } else if (e.ctrlKey && e.shiftKey && e.key === 'S') {
        e.preventDefault();
        skipPage();
    }
});

// Init
loadStats();
loadQueue();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return LABEL_HTML


def main():
    parser = argparse.ArgumentParser(description="Statement labeling UI")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-model", action="store_true", help="Skip model loading (empty pre-fills)")
    args = parser.parse_args()

    global _use_model
    if args.no_model:
        _use_model = False
        print("Model pre-fill disabled (--no-model)")

    print(f"Starting labeling UI on http://localhost:{args.port}")
    print(f"Pages dir: {PAGES_DIR}")
    print(f"Labels dir: {LABELS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
