"""Lightweight date extraction from fact text (no external deps, standalone).

Used to derive a fact's valid_from at ingestion time so temporal recall
('what was true before T') works without manual backdating. Heuristic but
covers the common cases produced by LLM extraction: an explicit 4-digit year,
an ISO date, a month+year, or a relative quarter ("early 2025", "Q1 2024").
"""

import re
from datetime import datetime, timezone
from typing import Optional

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS.keys()) + r")\s+(19|20)\d{2}\b", re.IGNORECASE)
_QUARTER_RE = re.compile(r"\bQ([1-4])\s+(19|20)\d{2}\b", re.IGNORECASE)


def extract_fact_date(text: str) -> Optional[datetime]:
    """Return a UTC datetime extracted from the fact text, or None if no date found."""
    if not text:
        return None
    low = text.lower()

    m = _ISO_RE.search(text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d, tzinfo=timezone.utc)
        except ValueError:
            pass

    m = _MONTH_RE.search(low)
    if m:
        mo = _MONTHS[m.group(1).lower()]
        y = int(m.group(2))
        return datetime(y, mo, 1, tzinfo=timezone.utc)

    m = _QUARTER_RE.search(low)
    if m:
        q, y = int(m.group(1)), int(m.group(2))
        mo = (q - 1) * 3 + 1
        return datetime(y, mo, 1, tzinfo=timezone.utc)

    m = _YEAR_RE.search(text)
    if m:
        y = int(m.group(0))
        # only treat as a valid year if plausibly a date context (not a random number)
        if 1900 <= y <= 2100:
            return datetime(y, 1, 1, tzinfo=timezone.utc)

    return None


__all__ = ["extract_fact_date"]
