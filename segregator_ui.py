#!/usr/bin/env python3
"""
Standalone UI to manually test the PDF-counting PLAN phase of the segregator.

Drag in several PDFs → see per-PDF page counts, total pages, and how long the
count took (should be milliseconds — it reads PDF metadata, not pixels).

Run:  python3 segregator_ui.py     then open http://127.0.0.1:8077
"""
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from pdf_segregator import plan_uploads, segregate_and_analyze, MAX_PARSE_WORKERS

app = FastAPI(title="PDF Segregator — Count + Analyze Tester")


@app.post("/count")
async def count(files: list[UploadFile] = File(...)):
    """PLAN phase only: page-count each uploaded PDF and report timing."""
    pdfs = [(f.filename, await f.read()) for f in files]

    t0 = time.time()
    plan = plan_uploads(pdfs)
    elapsed_ms = (time.time() - t0) * 1000

    total_pages = sum(p.pages for p in plan)
    workers = max(1, min(len(pdfs), MAX_PARSE_WORKERS))
    return {
        "pdfs": [
            {"pdf_idx": p.pdf_idx, "filename": p.filename,
             "pages": p.pages, "size_kb": round(p.nbytes / 1024),
             "bank_name": p.bank_name}
            for p in plan
        ],
        "pdf_count": len(plan),
        "total_pages": total_pages,
        "count_ms": round(elapsed_ms, 1),
        "planned_workers": workers,
        "budget_ok": elapsed_ms < 2000,
    }


@app.post("/analyze")
def analyze(files: list[UploadFile] = File(...)):
    """Full segregator: PLAN → parallel parse → provenance stamp → month regroup.

    Sync def so FastAPI runs the CPU/GPU-bound parse in its threadpool rather
    than blocking the event loop (the /v1/parse async-blocking lesson).
    """
    pdfs = [(f.filename, f.file.read()) for f in files]
    res = segregate_and_analyze(pdfs)
    return {
        "plan": [
            {"pdf_idx": p.pdf_idx, "filename": p.filename, "pages": p.pages,
             "bank_name": p.bank_name, "error": res.errors.get(p.pdf_idx)}
            for p in res.plan
        ],
        "timings_ms": res.timings_ms,
        "summary": res.summary,
        "transactions": res.transactions,
        "errors": res.errors,
        "reconciliation": res.reconciliation,
        "verdicts": res.verdicts,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PDF Count Tester</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f7; color: #1d1d1f; }
  .container { max-width: 760px; margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 24px; margin-bottom: 4px; }
  .subtitle { color: #86868b; font-size: 14px; margin-bottom: 24px; }
  .upload-area { background: #fff; border: 2px dashed #ccc; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: border-color 0.2s; }
  .upload-area:hover, .upload-area.dragover { border-color: #007aff; background: #f0f7ff; }
  .upload-area input { display: none; }
  .upload-area .icon { font-size: 40px; }
  .file-list { margin-top: 12px; font-size: 13px; color: #555; }
  .file-list span { background: #e8f0fe; padding: 3px 10px; border-radius: 12px; margin: 2px; display: inline-block; }
  .btn { background: #007aff; color: #fff; border: none; padding: 12px 28px; border-radius: 8px; font-size: 16px; cursor: pointer; margin-top: 16px; }
  .btn:disabled { background: #ccc; }
  .btn:hover:not(:disabled) { background: #0056b3; }
  .center { text-align: center; }
  #out { margin-top: 24px; display: none; }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
  .kpi { background: #fff; border-radius: 10px; padding: 16px; text-align: center; }
  .kpi .v { font-size: 26px; font-weight: 700; color: #007aff; }
  .kpi .l { font-size: 11px; color: #86868b; text-transform: uppercase; letter-spacing: .5px; margin-top: 4px; }
  .kpi .v.bad { color: #ff3b30; }
  table { width: 100%; background: #fff; border-radius: 12px; overflow: hidden; border-collapse: collapse; }
  th { background: #f8f8f8; text-align: left; padding: 10px 14px; font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: .5px; }
  td { padding: 10px 14px; border-top: 1px solid #f0f0f0; font-size: 14px; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .badge { display:inline-block; padding:3px 10px; border-radius:6px; font-size:12px; font-weight:600; }
  .badge.ok { background:#e7f7ec; color:#1d8a45; } .badge.bad { background:#fdebea; color:#c0382b; }
  .badge.na { background:#fff3e0; color:#b26a00; }
</style></head><body>
<div class="container">
  <h1>PDF Count Tester</h1>
  <p class="subtitle">PLAN phase only — counts pages from PDF metadata (no rendering). Should take milliseconds.</p>

  <div class="upload-area" id="dz">
    <div class="icon">&#128209;</div>
    <p id="lbl">Drop multiple PDFs here or click to select</p>
    <input type="file" id="fi" accept=".pdf" multiple>
    <div class="file-list" id="fl"></div>
  </div>
  <div class="center">
    <button class="btn" id="go" disabled>Count Pages</button>
    <button class="btn" id="goA" disabled style="background:#34c759;">Analyze</button>
  </div>

  <div id="out">
    <div class="kpis">
      <div class="kpi"><div class="v" id="kPdfs">—</div><div class="l">PDFs</div></div>
      <div class="kpi"><div class="v" id="kPages">—</div><div class="l">Total Pages</div></div>
      <div class="kpi"><div class="v" id="kMs">—</div><div class="l">Count Time</div></div>
      <div class="kpi"><div class="v" id="kWorkers">—</div><div class="l">Planned Workers</div></div>
    </div>
    <p style="margin-bottom:10px;" id="verdict"></p>
    <table>
      <thead><tr><th>#</th><th>Filename</th><th>Bank</th><th style="text-align:right">Pages</th><th style="text-align:right">Size</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>

  <div id="analysis" style="display:none;margin-top:28px;">
    <h2 style="font-size:18px;margin-bottom:12px;">Analysis</h2>
    <div class="kpis">
      <div class="kpi"><div class="v" id="aTxns">—</div><div class="l">Transactions</div></div>
      <div class="kpi"><div class="v" id="aParse">—</div><div class="l">Parse Time</div></div>
      <div class="kpi"><div class="v" id="aWorkers">—</div><div class="l">Workers</div></div>
      <div class="kpi"><div class="v" id="aMonths">—</div><div class="l">Months</div></div>
    </div>
    <p id="aErrors" style="margin-bottom:12px;"></p>

    <h3 style="font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:#888;margin:16px 0 8px;">Trust Verdict — per statement</h3>
    <p id="verdictSummary" style="margin-bottom:10px;"></p>
    <table>
      <thead><tr><th>PDF</th><th>Bank</th><th>Verdict</th><th>Running-balance</th><th>Summary-box</th></tr></thead>
      <tbody id="verdictRows"></tbody>
    </table>

    <h3 style="font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:#888;margin:20px 0 8px;">Accuracy — Running-Balance Reconciliation</h3>
    <p id="reconSummary" style="margin-bottom:10px;"></p>
    <table id="reconPdfTable">
      <thead><tr><th>PDF</th><th>Bank</th><th style="text-align:right">Rows Checked</th><th style="text-align:right">Reconcile</th><th style="text-align:right">No Balance</th><th style="text-align:right">Flagged</th></tr></thead>
      <tbody id="reconPdfRows"></tbody>
    </table>
    <h4 id="flagHdr" style="font-size:12px;color:#c0382b;margin:14px 0 6px;display:none;">Flagged rows (balance doesn't match amount)</h4>
    <table id="flagTable" style="display:none;">
      <thead><tr><th>page_uid</th><th>row</th><th>date</th><th>description</th><th style="text-align:right">signed amt</th><th style="text-align:right">prev bal</th><th style="text-align:right">balance</th><th style="text-align:right">expected</th><th style="text-align:right">Δ</th></tr></thead>
      <tbody id="flagRows"></tbody>
    </table>

    <h3 style="font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:#888;margin:20px 0 8px;">Per-Month Buckets</h3>
    <table>
      <thead><tr><th>Month</th><th style="text-align:right">Income</th><th style="text-align:right">Expenses</th><th style="text-align:right">Net</th><th style="text-align:right">Txns</th></tr></thead>
      <tbody id="monthRows"></tbody>
    </table>

    <h3 style="font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:#888;margin:20px 0 8px;">Transactions (with provenance)</h3>
    <table>
      <thead><tr><th>page_uid</th><th>row</th><th>month</th><th>date</th><th>source</th><th>description</th><th style="text-align:right">amount</th><th>type</th></tr></thead>
      <tbody id="txnRows"></tbody>
    </table>
  </div>
</div>
<script>
const dz=document.getElementById('dz'), fi=document.getElementById('fi'), go=document.getElementById('go'), goA=document.getElementById('goA');
let files=[];
['dragover','dragenter'].forEach(e=>dz.addEventListener(e,ev=>{ev.preventDefault();dz.classList.add('dragover');}));
['dragleave','drop'].forEach(e=>dz.addEventListener(e,ev=>{ev.preventDefault();dz.classList.remove('dragover');}));
dz.addEventListener('click',()=>fi.click());
dz.addEventListener('drop',ev=>{setFiles(ev.dataTransfer.files);});
fi.addEventListener('change',()=>setFiles(fi.files));
function setFiles(fl){
  files=[...fl].filter(f=>f.name.toLowerCase().endsWith('.pdf'));
  document.getElementById('fl').innerHTML=files.map(f=>`<span>${f.name}</span>`).join('');
  document.getElementById('lbl').textContent=files.length?`${files.length} PDF(s) selected`:'Drop multiple PDFs here or click to select';
  go.disabled=!files.length; goA.disabled=!files.length;
}
function money(v){ return (v<0?'-':'')+'$'+Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); }
go.addEventListener('click',async()=>{
  go.disabled=true; go.textContent='Counting…';
  const fd=new FormData(); files.forEach(f=>fd.append('files',f));
  try{
    const r=await fetch('count',{method:'POST',body:fd});
    const d=await r.json();
    document.getElementById('kPdfs').textContent=d.pdf_count;
    document.getElementById('kPages').textContent=d.total_pages;
    const ms=document.getElementById('kMs'); ms.textContent=d.count_ms+' ms'; ms.classList.toggle('bad',!d.budget_ok);
    document.getElementById('kWorkers').textContent=d.planned_workers;
    document.getElementById('verdict').innerHTML = d.budget_ok
      ? `<span class="badge ok">Under 2s budget &#10003;</span> &mdash; ${(d.count_ms/d.total_pages).toFixed(2)} ms/page`
      : `<span class="badge bad">Over 2s budget</span>`;
    document.getElementById('rows').innerHTML=d.pdfs.map(p=>{
      const bank=(p.bank_name&&p.bank_name!=='unknown')?p.bank_name:'<span style="color:#bbb">unknown</span>';
      return `<tr><td>${p.pdf_idx}</td><td>${p.filename}</td><td>${bank}</td><td class="num">${p.pages}</td><td class="num">${p.size_kb} KB</td></tr>`;
    }).join('');
    document.getElementById('out').style.display='block';
  }catch(e){ alert('Error: '+e.message); }
  go.disabled=false; go.textContent='Count Pages';
});

goA.addEventListener('click',async()=>{
  goA.disabled=true; goA.textContent='Analyzing… (first run loads the model)';
  const fd=new FormData(); files.forEach(f=>fd.append('files',f));
  try{
    const r=await fetch('analyze',{method:'POST',body:fd});
    const d=await r.json();
    const t=d.timings_ms||{}, s=d.summary||{};
    document.getElementById('aTxns').textContent=(d.transactions||[]).length;
    document.getElementById('aParse').textContent=(t.parse||0).toFixed(0)+' ms';
    document.getElementById('aWorkers').textContent=t.workers||'—';
    document.getElementById('aMonths').textContent=(s.months||[]).length;

    const errKeys=Object.keys(d.errors||{});
    document.getElementById('aErrors').innerHTML = errKeys.length
      ? `<span class="badge bad">${errKeys.length} PDF(s) failed</span> ` +
        errKeys.map(k=>`#${k}: ${d.errors[k]}`).join('; ')
      : `<span class="badge ok">all PDFs parsed &#10003;</span> plan ${(t.plan||0).toFixed(0)}ms · regroup ${(t.regroup||0).toFixed(1)}ms · total ${(t.total||0).toFixed(0)}ms`;

    // Trust verdicts (per statement)
    const verds=d.verdicts||{};
    const fnames0={}, fbanks0={}; (d.plan||[]).forEach(p=>{fnames0[p.pdf_idx]=p.filename; fbanks0[p.pdf_idx]=p.bank_name;});
    const vColor={reconciled:'#34c759', needs_review:'#ff3b30', unverified:'#ff9800'};
    const counts={reconciled:0, needs_review:0, unverified:0};
    Object.values(verds).forEach(v=>{ counts[v.verdict]=(counts[v.verdict]||0)+1; });
    document.getElementById('verdictSummary').innerHTML =
      `<span class="badge ok">${counts.reconciled||0} reconciled</span> `+
      `<span class="badge bad">${counts.needs_review||0} needs review</span> `+
      `<span class="badge na">${counts.unverified||0} unverified</span>`;
    function anchorCell(a){
      if(!a) return '—';
      const c={pass:'#34c759', fail:'#ff3b30', 'n/a':'#bbb'}[a.status]||'#888';
      let detail='';
      if(a.rate!=null) detail=` ${(a.rate*100).toFixed(1)}% (cov ${(a.coverage*100).toFixed(0)}%)`;
      else if(a.reason && a.reason!=='ok' && a.reason!=='no_totals') detail=` ${a.reason.replace('failed:','')}`;
      return `<span style="color:${c};font-weight:600">${a.status}</span><span style="color:#888;font-size:12px">${detail}</span>`;
    }
    document.getElementById('verdictRows').innerHTML=Object.entries(verds).map(([idx,v])=>{
      const bank=(fbanks0[idx]&&fbanks0[idx]!=='unknown')?fbanks0[idx]:'<span style="color:#bbb">unknown</span>';
      const vc=vColor[v.verdict]||'#888';
      return `<tr><td>${idx}: ${(fnames0[idx]||'').slice(0,20)}</td><td>${bank}</td>`+
             `<td><b style="color:${vc}">${v.verdict.replace('_',' ')}</b></td>`+
             `<td>${anchorCell(v.anchors&&v.anchors.running_balance)}</td>`+
             `<td>${anchorCell(v.anchors&&v.anchors.summary_box)}</td></tr>`;
    }).join('') || '<tr><td colspan="5" style="color:#999">verification disabled</td></tr>';

    // Reconciliation
    const rec=d.reconciliation||{};
    const rate=rec.overall_rate;
    const pct=(rate==null)?'—':(rate*100).toFixed(1)+'%';
    const cls=(rate==null)?'na':(rate>=0.99?'ok':(rate>=0.9?'na':'bad'));
    document.getElementById('reconSummary').innerHTML =
      `<span class="badge ${cls==='ok'?'ok':(cls==='bad'?'bad':'na')}">${pct} of rows reconcile</span> `+
      `${rec.total_ok||0}/${rec.total_checks||0} balance steps match · ${(rec.flagged||[]).length} flagged`;
    const fnames={}, fbanks={}; (d.plan||[]).forEach(p=>{fnames[p.pdf_idx]=p.filename; fbanks[p.pdf_idx]=p.bank_name;});
    document.getElementById('reconPdfRows').innerHTML=Object.entries(rec.per_pdf||{}).map(([idx,p])=>{
      const r=p.rate==null?'—':(p.rate*100).toFixed(1)+'%';
      const col=p.rate==null?'#888':(p.rate>=0.99?'#34c759':(p.rate>=0.9?'#ff9800':'#ff3b30'));
      const bank=(fbanks[idx]&&fbanks[idx]!=='unknown')?fbanks[idx]:'<span style="color:#bbb">unknown</span>';
      return `<tr><td>${idx}: ${(fnames[idx]||'').slice(0,22)}</td><td>${bank}</td><td class="num">${p.checks}</td>`+
             `<td class="num" style="color:${col};font-weight:600">${r}</td><td class="num">${p.no_balance}</td>`+
             `<td class="num">${p.flagged}</td></tr>`;
    }).join('');
    const flags=rec.flagged||[];
    document.getElementById('flagHdr').style.display=flags.length?'block':'none';
    document.getElementById('flagTable').style.display=flags.length?'table':'none';
    document.getElementById('flagRows').innerHTML=flags.slice(0,200).map(f=>
      `<tr><td><b>${f.page_uid}</b></td><td>${f.row}</td><td>${f.date||''}</td><td>${(f.description||'')}</td>`+
      `<td class="num">${money(f.signed_amount)}</td><td class="num">${money(f.prev_balance)}</td>`+
      `<td class="num">${money(f.balance_after)}</td><td class="num">${money(f.expected)}</td>`+
      `<td class="num" style="color:#ff3b30;font-weight:600">${money(f.delta)}</td></tr>`).join('');

    const inc=s.monthly_income||{}, exp=s.monthly_expenses||{}, cnt=s.monthly_txn_count||{};
    document.getElementById('monthRows').innerHTML=(s.months||[]).map(m=>{
      const net=(inc[m]||0)-Math.abs(exp[m]||0);
      return `<tr><td>${m}</td><td class="num">${money(inc[m]||0)}</td><td class="num">${money(-Math.abs(exp[m]||0))}</td>`+
             `<td class="num" style="color:${net>=0?'#34c759':'#ff3b30'}">${money(net)}</td><td class="num">${cnt[m]||0}</td></tr>`;
    }).join('') || '<tr><td colspan="5" style="color:#999">no dated transactions</td></tr>';

    document.getElementById('txnRows').innerHTML=(d.transactions||[]).slice(0,500).map(x=>
      `<tr><td><b>${x.page_uid}</b></td><td>${x.row}</td><td>${x.month_key}</td><td>${x.date||''}</td>`+
      `<td>${(x.source_file||'').slice(0,16)}</td><td>${(x.description||'').slice(0,42)}</td>`+
      `<td class="num ${x.amount<0?'':''}" style="color:${x.type==='credit'?'#34c759':'#ff3b30'}">${money(x.type==='credit'?x.amount:-Math.abs(x.amount))}</td><td>${x.type||''}</td></tr>`).join('');
    document.getElementById('analysis').style.display='block';
    document.getElementById('analysis').scrollIntoView({behavior:'smooth'});
  }catch(e){ alert('Error: '+e.message); }
  goA.disabled=false; goA.textContent='Analyze';
});
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    print("PDF Count Tester → http://127.0.0.1:8077")
    uvicorn.run(app, host="127.0.0.1", port=8077, log_level="warning")
