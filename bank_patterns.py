"""Free bank-name detection (no ML / no API), shared by predict.py and pdf_segregator.py.

Single source of truth for BANK_PATTERNS so the segregator and the core parser
never drift. Two zero-cost strategies, tried in order:

  1. BANK_PATTERNS  — keyword match against page-1 text → canonical name (known banks)
  2. header heuristic — "<Capitalized phrase> Bank/Credit Union" or a domain-derived
     name (catches FourLeaf-style institutions not in the pattern list)

Kept dependency-light on purpose: importing this must NOT pull in torch/transformers,
so the PLAN phase (page counting + bank detection) stays fast and model-free.
"""
import re
from typing import Optional

# Common bank identifiers for feedback grouping + detection.
BANK_PATTERNS = {
    "chase": "Chase", "jpmorgan": "Chase",
    "wells fargo": "Wells Fargo", "wellsfargo": "Wells Fargo",
    "bank of america": "Bank of America", "bankofamerica": "Bank of America",
    "citibank": "Citibank", "citi ": "Citibank",
    "us bank": "US Bank", "u.s. bank": "US Bank",
    "pnc": "PNC", "td bank": "TD Bank",
    "capital one": "Capital One", "truist": "Truist",
    "regions": "Regions", "keybank": "KeyBank",
    "comerica": "Comerica", "citizens": "Citizens",
    "navy federal": "Navy Federal", "usaa": "USAA",
    "huntington": "Huntington", "m&t bank": "M&T Bank",
    "fifth third": "Fifth Third", "zions": "Zions",
    "bbva": "BBVA", "bmo": "BMO",
}

# Suffixes stripped from a domain SLD before recovering the institution name.
# Order matters: longer/more-specific first so "fcu" wins over "cu".
_DOMAIN_SUFFIXES = (
    "federalcreditunion", "creditunion", "fcu", "cu",
    "bankonline", "onlinebanking", "banking", "bank", "fsb",
    "financial", "online", "na",
)

# Generic words that, alone, are not a real institution name.
_GENERIC = {"the", "your", "online", "member", "statement", "account", "national"}


def _from_patterns(low_text: str) -> Optional[str]:
    for pattern, name in BANK_PATTERNS.items():
        if pattern in low_text:
            return name
    return None


def _from_header(text: str) -> Optional[str]:
    """Best-effort institution name from the page-1 header region.

    Restricted to the top of the page (where the bank's own name/domain lives)
    to avoid grabbing merchant URLs from transaction rows further down.
    """
    head = text[:900]  # header only — keeps merchant URLs in the txn table out

    # (a) "<Capitalized phrase> Bank | Credit Union | Savings Bank | Financial"
    m = re.search(
        r"([A-Z][A-Za-z&.'\-]+(?:\s+[A-Z][A-Za-z&.'\-]+){0,3})\s+"
        r"(Federal Credit Union|Credit Union|Savings Bank|Bank|Financial)\b",
        head,
    )
    if m:
        phrase = m.group(1).strip()
        if phrase.lower() not in _GENERIC:
            return f"{phrase} {m.group(2)}".strip()

    # (b) domain → SLD → strip banking suffix → recover nicely-cased form from text.
    # Reject digit-bearing slugs: real bank domains are alphabetic (digit ones
    # like Fifth Third's 53.com are already covered by BANK_PATTERNS); a slug with
    # digits is almost always a merchant descriptor, not the institution.
    m = re.search(r"\b([a-z][a-z\-]{2,})\.(?:com|org|net|us|bank)\b", head.lower())
    if m:
        slug = m.group(1)
        for suf in _DOMAIN_SUFFIXES:
            if slug.endswith(suf) and len(slug) > len(suf) + 1:
                slug = slug[: -len(suf)]
                break
        if len(slug) >= 3 and slug not in _GENERIC:
            # Prefer a capitalized standalone occurrence (the wordmark "FourLeaf")
            # over the lowercase domain match ("fourleaffcu.com").
            for mm in re.finditer(re.escape(slug), head, re.IGNORECASE):
                token = head[mm.start(): mm.start() + len(slug)]
                if token[:1].isupper():
                    return token
            return slug.title()

    return None


def detect_bank(text: str) -> str:
    """Return a bank/institution name from page-1 text, or 'unknown'.

    Free tiers only: patterns (canonical) → header phrase → domain-derived.
    Returns 'unknown' when none match — the caller escalates those to Claude
    (resolve_bank_with_claude) for the guaranteed-name fallback.
    """
    if not text:
        return "unknown"
    return _from_patterns(text.lower()) or _from_header(text) or "unknown"
