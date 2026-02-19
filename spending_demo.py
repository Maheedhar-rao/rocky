#!/usr/bin/env python3
"""
Spending analysis demo — parse bank statement PDFs and generate
a self-contained HTML report with charts.

Usage:
    python spending_demo.py data/pdfs/*.pdf
    python spending_demo.py --dir data/pdfs/
    python spending_demo.py statement_jan.pdf statement_feb.pdf --output report.html
    python spending_demo.py data/pdfs/*.pdf --json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parent

# --- Spending Categories (ordered by specificity, first match wins) ---
CATEGORIES = [
    ("Payroll/Income", [
        "direct deposit", "payroll", "salary", "wages", "direct dep",
        "ach credit payroll", "employer", "pay id",
    ], True),
    ("Rent/Mortgage", [
        "rent", "mortgage", "landlord", "property mgmt", "hoa", "housing",
        "apartment", "lease",
    ], False),
    ("Utilities", [
        "electric", "gas co", "water", "internet", "phone", "comcast", "at&t",
        "att uverse", "t-mobile", "verizon", "spectrum", "xfinity", "utility",
        "power", "sewer", "trash", "fpl ", "duke energy", "pg&e",
    ], False),
    ("Food/Grocery", [
        "walmart", "kroger", "sprouts", "whole foods", "trader joe", "aldi",
        "costco", "sam's club", "publix", "safeway", "heb", "grocery",
        "target", "food lion", "winn dixie", "piggly", "albertsons",
        "food mart", "market basket",
    ], False),
    ("Restaurants/Dining", [
        "restaurant", "mcdonald", "starbucks", "chick-fil", "wendy",
        "taco bell", "chipotle", "doordash", "uber eats", "grubhub",
        "pizza", "diner", "cafe", "dunkin", "panera", "subway", "domino",
        "dining", "burger", "popeyes", "kfc", "panda express",
    ], False),
    ("Subscriptions", [
        "netflix", "spotify", "google play", "apple.com", "hulu", "disney+",
        "amazon prime", "youtube", "adobe", "microsoft", "dropbox", "icloud",
        "chatgpt", "openai",
    ], False),
    ("Transfers", [
        "zelle", "venmo", "cashapp", "cash app", "wire", "transfer",
        "p2p", "money sent", "paypal",
    ], False),
    ("Loan Payments", [
        "lending", "funding", "loan", "student loan", "auto loan",
        "sba", "kabbage", "ondeck", "bluevine", "credibly", "pipe cap",
    ], False),
    ("Fees", [
        "overdraft", "nsf", "service charge", "maintenance fee",
        "atm fee", "monthly fee", "insufficient", "returned item",
        "od fee", "service fee", "monthly service",
    ], False),
]
DEFAULT_CATEGORY = "Other"


def categorize_transaction(txn: dict) -> str:
    """Classify a transaction by keyword matching on description."""
    desc_lower = txn["description"].lower()
    for cat_name, keywords, _ in CATEGORIES:
        if any(kw in desc_lower for kw in keywords):
            return cat_name
    return DEFAULT_CATEGORY


def parse_date(date_str: str, year_hint: int = None, statement_end_month: int = None) -> Optional[datetime]:
    """Parse transaction date string into a datetime.

    For cross-year statements (e.g., Dec 2025 - Jan 2026), year_hint is the END
    year (2026) and statement_end_month is the end month (1 = January). Any
    transaction month > statement_end_month belongs to the previous year.
    """
    if not date_str or not date_str.strip():
        return None

    date_str = date_str.strip().rstrip(".")
    year = year_hint or datetime.now().year

    patterns = [
        (r"^(\d{1,2})/(\d{1,2})/(\d{4})$", lambda m: datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))),
        (r"^(\d{1,2})/(\d{1,2})/(\d{2})$", lambda m: datetime(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))),
        (r"^(\d{1,2})/(\d{1,2})$", lambda m: datetime(year, int(m.group(1)), int(m.group(2)))),
        (r"^(\d{1,2})-(\d{1,2})-(\d{4})$", lambda m: datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))),
        (r"^(\d{1,2})-(\d{1,2})-(\d{2})$", lambda m: datetime(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))),
        (r"^(\d{1,2})-(\d{1,2})$", lambda m: datetime(year, int(m.group(1)), int(m.group(2)))),
    ]

    dt = None
    for pattern, builder in patterns:
        m = re.match(pattern, date_str)
        if m:
            try:
                dt = builder(m)
                break
            except ValueError:
                continue

    if dt is None:
        # Month name formats: "Jan 15", "January 15, 2025"
        for fmt in ("%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y", "%b %d", "%B %d"):
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.year == 1900:
                    dt = dt.replace(year=year)
                break
            except ValueError:
                continue

    if dt is None:
        return None

    # Cross-year correction: if statement ends in Jan (month 1) but this date
    # is in Dec (month 12), it belongs to the previous year.
    if statement_end_month and dt.month > statement_end_month and dt.year == year:
        dt = dt.replace(year=year - 1)

    return dt


def infer_year_from_filename(pdf_path: Path) -> Optional[int]:
    """Try to extract a year (2020-2030) from the PDF filename."""
    match = re.search(r"20[2-3]\d", pdf_path.stem)
    if match:
        return int(match.group())
    return None


def infer_year_from_pdf(pdf_bytes: bytes) -> tuple:
    """Extract the statement year and end month from PDF content.

    Returns (year, end_month) tuple. end_month is used to handle cross-year
    statements (e.g., Dec 2025 - Jan 2026 → year=2026, end_month=1).
    """
    try:
        import pdfplumber
        import io
        from collections import Counter
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return None, None

            # Check first and last pages (statement period appears on both)
            page_indices = [0]
            if len(pdf.pages) > 1:
                page_indices.append(len(pdf.pages) - 1)

            years = []

            MONTH_MAP = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                "may": 5, "jun": 6, "jul": 7, "aug": 8,
                "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            }

            for pi in page_indices:
                text = pdf.pages[pi].extract_text() or ""

                # --- Priority 1: Date range END date ---
                # Numeric ranges: "12/20/2025 - 01/22/2026"
                m = re.search(
                    r"(\d{1,2})[/-]\d{1,2}[/-](?:20[2-3]\d|\d{2})"
                    r"\s*(?:through|thru|to|-|–|—)\s*"
                    r"(\d{1,2})[/-]\d{1,2}[/-](20[2-3]\d|\d{2})",
                    text, re.IGNORECASE,
                )
                if m:
                    end_month = int(m.group(2))
                    y = int(m.group(3))
                    y = y + 2000 if y < 100 else y
                    return y, end_month

                # Month-name ranges: "December 25, 2025 through January 28, 2026"
                m = re.search(
                    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?"
                    r"\s+\d{0,2},?\s*(?:20[2-3]\d)"
                    r"\s*(?:through|thru|to|-|–|—)\s*"
                    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?)"
                    r"\s+\d{0,2},?\s*(20[2-3]\d)",
                    text, re.IGNORECASE,
                )
                if m:
                    end_month_name = m.group(1)[:3].lower()
                    end_month = MONTH_MAP.get(end_month_name)
                    return int(m.group(2)), end_month

                # --- Priority 2: "through/ending/closing" keyword ---
                m = re.search(
                    r"(?:through|ending|closing|thru)\s+.*?(20[2-3]\d)",
                    text, re.IGNORECASE,
                )
                if m:
                    years.extend([int(m.group(1))] * 5)

                # Years from full dates: MM/DD/YYYY, MM-DD-YYYY
                for m in re.finditer(r"\d{1,2}[/-]\d{1,2}[/-](20[2-3]\d)", text):
                    years.append(int(m.group(1)))
                # Years from month name dates: "January 2025", "Dec 15, 2024"
                for m in re.finditer(
                    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?"
                    r"\s+(?:\d{1,2},?\s+)?(20[2-3]\d)",
                    text, re.IGNORECASE,
                ):
                    years.append(int(m.group(1)))

            if years:
                return Counter(years).most_common(1)[0][0], None
    except Exception:
        pass
    return None, None


def process_pdfs(pdf_paths: list) -> list:
    """Parse all PDFs and return enriched transactions with categories."""
    from predict import get_parser

    print("Loading model...")
    parser = get_parser()
    print(f"Model version: {parser.version}")

    all_transactions = []

    for i, pdf_path in enumerate(pdf_paths):
        print(f"  [{i+1}/{len(pdf_paths)}] Parsing {pdf_path.name}...")
        pdf_bytes = pdf_path.read_bytes()
        result = parser.parse_pdf(pdf_bytes)

        year_hint = infer_year_from_filename(pdf_path)
        end_month = None
        if not year_hint:
            year_hint, end_month = infer_year_from_pdf(pdf_bytes)

        for txn in result.transactions:
            dt = parse_date(txn["date"], year_hint, statement_end_month=end_month)
            category = categorize_transaction(txn)

            all_transactions.append({
                **txn,
                "parsed_date": dt,
                "month_key": dt.strftime("%Y-%m") if dt else "unknown",
                "category": category,
                "source_file": pdf_path.name,
            })

    all_transactions.sort(key=lambda t: t["parsed_date"] or datetime.min)
    return all_transactions


def compute_summary(transactions: list) -> dict:
    """Compute all summary statistics and chart data."""
    monthly_income = defaultdict(float)
    monthly_expenses = defaultdict(float)
    monthly_txn_count = defaultdict(int)
    category_totals = defaultdict(float)
    daily_balances = {}
    merchant_counts = defaultdict(lambda: {"count": 0, "total": 0.0})

    for txn in transactions:
        month = txn["month_key"]
        if month == "unknown":
            continue

        monthly_txn_count[month] += 1

        if txn["type"] == "credit":
            monthly_income[month] += txn["amount"]
        else:
            monthly_expenses[month] += txn["amount"]
            category_totals[txn["category"]] += txn["amount"]

        if txn["balance_after"] is not None and txn["parsed_date"]:
            date_key = txn["parsed_date"].strftime("%Y-%m-%d")
            daily_balances[date_key] = txn["balance_after"]

        if txn["type"] == "debit":
            merchant_key = txn["description"][:35].strip()
            if merchant_key:
                merchant_counts[merchant_key]["count"] += 1
                merchant_counts[merchant_key]["total"] += txn["amount"]

    all_months = sorted(set(list(monthly_income.keys()) + list(monthly_expenses.keys())))

    top_merchants = sorted(
        merchant_counts.items(),
        key=lambda x: x[1]["total"],
        reverse=True,
    )[:15]

    sorted_balances = sorted(daily_balances.items())

    total_income = sum(monthly_income.values())
    total_expenses = sum(monthly_expenses.values())
    months_count = len(all_months) or 1

    return {
        "months": all_months,
        "monthly_income": {m: round(monthly_income[m], 2) for m in all_months},
        "monthly_expenses": {m: round(monthly_expenses[m], 2) for m in all_months},
        "monthly_txn_count": {m: monthly_txn_count[m] for m in all_months},
        "category_totals": {k: round(v, 2) for k, v in sorted(category_totals.items(), key=lambda x: -x[1])},
        "daily_balances": sorted_balances,
        "top_merchants": [(k, v) for k, v in top_merchants],
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "avg_monthly_spend": round(total_expenses / months_count, 2),
        "total_transactions": len([t for t in transactions if t["month_key"] != "unknown"]),
        "months_analyzed": months_count,
        "net_cashflow": round(total_income - total_expenses, 2),
        "flagged_transactions": sum(
            1 for t in transactions
            if t.get("confidence", 1.0) < 0.75 and t["month_key"] != "unknown"
        ),
    }


def _fmt_money(val: float) -> str:
    """Format number as $X,XXX.XX."""
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val):,.2f}"


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def generate_html(summary: dict, transactions: list) -> str:
    """Generate self-contained HTML report with Chart.js charts."""

    # Month labels for charts
    month_labels = []
    for m in summary["months"]:
        try:
            dt = datetime.strptime(m, "%Y-%m")
            month_labels.append(dt.strftime("%b %Y"))
        except ValueError:
            month_labels.append(m)

    income_data = [summary["monthly_income"].get(m, 0) for m in summary["months"]]
    expense_data = [summary["monthly_expenses"].get(m, 0) for m in summary["months"]]

    cat_labels = list(summary["category_totals"].keys())
    cat_data = list(summary["category_totals"].values())

    balance_labels = [b[0] for b in summary["daily_balances"]]
    balance_data = [b[1] for b in summary["daily_balances"]]
    show_balance = len(balance_data) >= 5

    # Monthly breakdown table rows
    monthly_rows = ""
    for m, label in zip(summary["months"], month_labels):
        inc = summary["monthly_income"].get(m, 0)
        exp = summary["monthly_expenses"].get(m, 0)
        net = inc - exp
        txns = summary["monthly_txn_count"].get(m, 0)
        net_class = "positive" if net >= 0 else "negative"
        monthly_rows += f"""<tr>
            <td>{_html_escape(label)}</td>
            <td class="money">{_fmt_money(inc)}</td>
            <td class="money">{_fmt_money(exp)}</td>
            <td class="money {net_class}">{_fmt_money(net)}</td>
            <td>{txns}</td>
        </tr>\n"""

    # Top merchants table rows
    merchant_rows = ""
    for name, data in summary["top_merchants"]:
        merchant_rows += f"""<tr>
            <td>{_html_escape(name)}</td>
            <td>{data['count']}</td>
            <td class="money">{_fmt_money(data['total'])}</td>
        </tr>\n"""

    # Category colors
    colors = [
        "#ff6384", "#36a2eb", "#ffce56", "#4bc0c0", "#9966ff",
        "#ff9f40", "#c9cbcf", "#e7e9ed", "#7bc043", "#f37735",
    ]

    net_class = "positive" if summary["net_cashflow"] >= 0 else "negative"
    gen_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    chart_data = json.dumps({
        "monthLabels": month_labels,
        "incomeData": income_data,
        "expenseData": expense_data,
        "catLabels": cat_labels,
        "catData": cat_data,
        "catColors": colors[:len(cat_labels)],
        "balanceLabels": balance_labels,
        "balanceData": balance_data,
        "showBalance": show_balance,
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spending Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f7;
            color: #1d1d1f;
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}

        header {{
            text-align: center;
            margin-bottom: 32px;
        }}
        header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .subtitle {{ color: #86868b; font-size: 14px; }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            background: #fff;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .card .label {{
            font-size: 12px;
            color: #86868b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .card .value {{
            font-size: 22px;
            font-weight: 700;
        }}
        .positive {{ color: #34c759; }}
        .negative {{ color: #ff3b30; }}

        .chart-section {{
            background: #fff;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .chart-section h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
        }}
        .chart-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}
        @media (max-width: 768px) {{
            .chart-row {{ grid-template-columns: 1fr; }}
        }}

        .table-section {{
            background: #fff;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            overflow-x: auto;
        }}
        .table-section h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 2px solid #e5e5ea;
            color: #86868b;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #f2f2f7;
        }}
        tr:hover {{ background: #f9f9fb; }}
        .money {{ font-variant-numeric: tabular-nums; text-align: right; }}
        th.money {{ text-align: right; }}

        footer {{
            text-align: center;
            color: #86868b;
            font-size: 12px;
            margin-top: 32px;
            padding-bottom: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Spending Analysis Report</h1>
            <p class="subtitle">Generated {gen_date} &middot; {summary['months_analyzed']} months analyzed &middot; {summary['total_transactions']} transactions</p>
        </header>

        <div class="summary-grid">
            <div class="card">
                <div class="label">Total Income</div>
                <div class="value positive">{_fmt_money(summary['total_income'])}</div>
            </div>
            <div class="card">
                <div class="label">Total Expenses</div>
                <div class="value negative">{_fmt_money(summary['total_expenses'])}</div>
            </div>
            <div class="card">
                <div class="label">Net Cashflow</div>
                <div class="value {net_class}">{_fmt_money(summary['net_cashflow'])}</div>
            </div>
            <div class="card">
                <div class="label">Avg Monthly Spend</div>
                <div class="value">{_fmt_money(summary['avg_monthly_spend'])}</div>
            </div>
            <div class="card">
                <div class="label">Transactions</div>
                <div class="value">{summary['total_transactions']:,}</div>
            </div>
            <div class="card">
                <div class="label">Months</div>
                <div class="value">{summary['months_analyzed']}</div>
            </div>
        </div>

        <div class="chart-section">
            <h2>Monthly Income vs Expenses</h2>
            <canvas id="monthlyChart" height="100"></canvas>
        </div>

        <div class="chart-row">
            <div class="chart-section">
                <h2>Spending by Category</h2>
                <canvas id="categoryChart"></canvas>
            </div>
            {"" if not show_balance else '''<div class="chart-section">
                <h2>Account Balance Trend</h2>
                <canvas id="balanceChart"></canvas>
            </div>'''}
        </div>

        <div class="table-section">
            <h2>Monthly Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Month</th>
                        <th class="money">Income</th>
                        <th class="money">Expenses</th>
                        <th class="money">Net</th>
                        <th>Txns</th>
                    </tr>
                </thead>
                <tbody>
                    {monthly_rows}
                </tbody>
            </table>
        </div>

        <div class="table-section">
            <h2>Top Merchants</h2>
            <table>
                <thead>
                    <tr>
                        <th>Merchant</th>
                        <th>Transactions</th>
                        <th class="money">Total Spent</th>
                    </tr>
                </thead>
                <tbody>
                    {merchant_rows}
                </tbody>
            </table>
        </div>

        <footer>
            Powered by LayoutLMv3 Statement Parser &middot; ML-based transaction extraction
        </footer>
    </div>

    <script>
        const DATA = {chart_data};

        // Monthly Income vs Expenses
        new Chart(document.getElementById('monthlyChart'), {{
            type: 'bar',
            data: {{
                labels: DATA.monthLabels,
                datasets: [
                    {{
                        label: 'Income',
                        data: DATA.incomeData,
                        backgroundColor: '#34c759',
                        borderRadius: 4,
                    }},
                    {{
                        label: 'Expenses',
                        data: DATA.expenseData,
                        backgroundColor: '#ff3b30',
                        borderRadius: 4,
                    }},
                ],
            }},
            options: {{
                responsive: true,
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: ctx => ctx.dataset.label + ': $' + ctx.parsed.y.toLocaleString(undefined, {{minimumFractionDigits: 2}})
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            callback: v => '$' + v.toLocaleString()
                        }}
                    }}
                }}
            }}
        }});

        // Spending by Category
        new Chart(document.getElementById('categoryChart'), {{
            type: 'doughnut',
            data: {{
                labels: DATA.catLabels,
                datasets: [{{
                    data: DATA.catData,
                    backgroundColor: DATA.catColors,
                    borderWidth: 2,
                    borderColor: '#fff',
                }}],
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ position: 'right', labels: {{ font: {{ size: 12 }} }} }},
                    tooltip: {{
                        callbacks: {{
                            label: ctx => {{
                                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                                const pct = ((ctx.parsed / total) * 100).toFixed(1);
                                return ctx.label + ': $' + ctx.parsed.toLocaleString(undefined, {{minimumFractionDigits: 2}}) + ' (' + pct + '%)';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Balance Trend
        if (DATA.showBalance) {{
            new Chart(document.getElementById('balanceChart'), {{
                type: 'line',
                data: {{
                    labels: DATA.balanceLabels,
                    datasets: [{{
                        label: 'Balance',
                        data: DATA.balanceData,
                        borderColor: '#007aff',
                        backgroundColor: 'rgba(0, 122, 255, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 1,
                    }}],
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        tooltip: {{
                            callbacks: {{
                                label: ctx => '$' + ctx.parsed.y.toLocaleString(undefined, {{minimumFractionDigits: 2}})
                            }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            ticks: {{
                                callback: v => '$' + v.toLocaleString()
                            }}
                        }},
                        x: {{
                            ticks: {{ maxTicksLimit: 12 }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Parse bank statements and generate spending analysis report"
    )
    parser.add_argument("pdfs", nargs="*", help="PDF file paths")
    parser.add_argument("--dir", help="Directory containing PDF files")
    parser.add_argument("--output", "-o", default="spending_report.html",
                        help="Output HTML file (default: spending_report.html)")
    parser.add_argument("--json", action="store_true",
                        help="Also output raw data as JSON")
    args = parser.parse_args()

    # Collect PDF paths
    pdf_paths = []
    if args.pdfs:
        for p in args.pdfs:
            pdf_paths.append(Path(p))
    if args.dir:
        d = Path(args.dir)
        pdf_paths.extend(sorted(d.glob("*.pdf")))
        pdf_paths.extend(sorted(d.glob("*.PDF")))

    pdf_paths = sorted(set(pdf_paths))

    if not pdf_paths:
        print("No PDF files found. Usage: python spending_demo.py *.pdf")
        sys.exit(1)

    # Filter to existing files
    pdf_paths = [p for p in pdf_paths if p.exists()]
    print(f"Found {len(pdf_paths)} PDF files\n")

    # Parse
    transactions = process_pdfs(pdf_paths)
    print(f"\nExtracted {len(transactions)} total transactions")

    # Summarize
    summary = compute_summary(transactions)

    print(f"\n{'='*50}")
    print(f"  Total income:      {_fmt_money(summary['total_income'])}")
    print(f"  Total expenses:    {_fmt_money(summary['total_expenses'])}")
    print(f"  Net cashflow:      {_fmt_money(summary['net_cashflow'])}")
    print(f"  Avg monthly spend: {_fmt_money(summary['avg_monthly_spend'])}")
    print(f"  Months analyzed:   {summary['months_analyzed']}")
    print(f"{'='*50}")

    # Generate HTML
    html = generate_html(summary, transactions)
    output_path = Path(args.output)
    output_path.write_text(html)
    print(f"\nReport saved to {output_path.resolve()}")

    if args.json:
        json_path = output_path.with_suffix(".json")
        json_data = {
            "summary": {k: v for k, v in summary.items()},
            "transactions": [
                {k: v for k, v in t.items() if k != "parsed_date"}
                for t in transactions
            ],
        }
        json_path.write_text(json.dumps(json_data, indent=2, default=str))
        print(f"JSON data saved to {json_path}")


if __name__ == "__main__":
    main()
