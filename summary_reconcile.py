"""
Summary-box reconciliation — the second verification anchor.

Bank statements print their own control totals (beginning/ending balance, total
deposits, total withdrawals). This cross-checks the EXTRACTED transactions against
those printed totals, so we can prove a statement is right (or flag it) even when
the bank prints no per-row running balance (e.g. Chase) — exactly where the
running-balance anchor goes blind.

Source-agnostic: the stated totals can come from a free regex over page-1 text or
from the existing Claude page-1 metadata (_extract_statement_metadata). This module
only does the math + verdict.

Three checks:
  1. balance equation:  beginning + sum(signed extracted) ≈ ending
  2. gross deposits:    sum(credits) ≈ stated_total_deposits
  3. gross withdrawals: sum(debits)  ≈ stated_total_withdrawals
Check 1 catches net errors (phantom rows, sign flips). Checks 2-3 catch gross
errors that net out (double-counted reversals) — the FourLeaf case.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SummaryVerdict:
    reconciled: bool
    checks: dict = field(default_factory=dict)   # name -> {extracted, stated, delta, ok}
    reason: str = ""


def _signed(txn) -> float:
    """Signed amount: credits +, debits -. Handles both signed amounts and abs+type."""
    amt = txn.get("amount", 0) or 0
    t = txn.get("type")
    if t == "credit":
        return abs(amt)
    if t == "debit":
        return -abs(amt)
    return float(amt)  # already signed


def reconcile_summary(
    transactions: list,
    beginning: Optional[float] = None,
    ending: Optional[float] = None,
    stated_total_deposits: Optional[float] = None,
    stated_total_withdrawals: Optional[float] = None,
    abs_tol: float = 0.02,
) -> SummaryVerdict:
    """Cross-check extracted transactions against the statement's printed totals.

    Any total left as None skips that check. A statement reconciles only if every
    check it CAN run passes. If no totals are available at all, it can't be judged
    here (reconciled=False, reason='no_totals') — that's a routing signal, not a
    claim the extraction is wrong.
    """
    credits = sum(_signed(t) for t in transactions if _signed(t) > 0)
    debits = -sum(_signed(t) for t in transactions if _signed(t) < 0)   # positive magnitude
    net = credits - debits

    checks = {}

    if beginning is not None and ending is not None:
        computed_end = round(beginning + net, 2)
        delta = round(computed_end - ending, 2)
        checks["balance_equation"] = {
            "extracted": computed_end, "stated": ending,
            "delta": delta, "ok": abs(delta) <= abs_tol,
        }

    if stated_total_deposits is not None:
        delta = round(credits - stated_total_deposits, 2)
        checks["gross_deposits"] = {
            "extracted": round(credits, 2), "stated": stated_total_deposits,
            "delta": delta, "ok": abs(delta) <= abs_tol,
        }

    if stated_total_withdrawals is not None:
        delta = round(debits - stated_total_withdrawals, 2)
        checks["gross_withdrawals"] = {
            "extracted": round(debits, 2), "stated": stated_total_withdrawals,
            "delta": delta, "ok": abs(delta) <= abs_tol,
        }

    if not checks:
        return SummaryVerdict(reconciled=False, checks={}, reason="no_totals")

    failed = [name for name, c in checks.items() if not c["ok"]]
    return SummaryVerdict(
        reconciled=(len(failed) == 0),
        checks=checks,
        reason="ok" if not failed else "failed:" + ",".join(failed),
    )
