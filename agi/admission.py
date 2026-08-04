"""E32 — Adaptive memory admission control (write-gate).

Reference: Adaptive Memory Admission Control (2603.04549). Gate at WRITE time so
bloat never enters the store, instead of relying only on read-time decay.

Score = w_n·novelty + w_i·importance + w_c·credibility, adjusted by capacity headroom.
Decision: STORE (>= store_threshold) | MERGE (mid band, near-dup exists) | DROP.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Any

from config import settings
from utils import log_atomic

# Tautologies / zero-information statements that should never occupy memory.
_LOW_VALUE_MARKERS = (
    "i am an ai", "as an ai", "hello", "hi there", "thanks", "thank you",
    "ok", "okay", "sure", "no problem", "you're welcome",
)

STORE_THRESHOLD = 0.45
MERGE_THRESHOLD = 0.25
CAPACITY_SOFT_LIMIT = 20000  # facts per user before headroom pressure applies


@dataclass
class AdmissionDecision:
    action: str            # STORE | MERGE | DROP
    score: float
    novelty: float
    importance: float
    credibility: float
    reason: str

    @property
    def admitted(self) -> bool:
        return self.action != "DROP"


def _novelty(candidates: Sequence[Any]) -> float:
    """1.0 when nothing similar exists; approaches 0 as a near-duplicate appears."""
    if not candidates:
        return 1.0
    top = max(float(getattr(c, "similarity", 0.0) or 0.0) for c in candidates)
    return max(0.0, min(1.0, 1.0 - top))


def _importance(fact_text: str) -> float:
    """Cheap heuristic importance in 0..1 (LLM salience judge refines later, E14)."""
    text = fact_text.strip()
    lc = text.lower()
    if not text:
        return 0.0
    if any(lc.startswith(m) or lc == m for m in _LOW_VALUE_MARKERS):
        return 0.05
    score = 0.4
    words = len(text.split())
    if words >= 5:
        score += 0.2
    if words >= 12:
        score += 0.1
    # Concrete signal: numbers, proper nouns, preference/identity verbs.
    if any(ch.isdigit() for ch in text):
        score += 0.1
    if any(w[:1].isupper() for w in text.split()[1:]):
        score += 0.1
    if any(k in lc for k in ("prefer", "always", "never", "name is", "works at",
                             "lives in", "birthday", "deadline", "must", "hates", "loves")):
        score += 0.2
    return max(0.0, min(1.0, score))


def _credibility(source_trust: float, provenance_kind: str) -> float:
    base = max(0.0, min(1.0, float(source_trust)))
    if provenance_kind == "implicit_inferred":
        base *= 0.6   # E30: down-weight model-inferred content
    return base


def _headroom(active_count: Optional[int]) -> float:
    if active_count is None:
        return 1.0
    if active_count < CAPACITY_SOFT_LIMIT:
        return 1.0
    over = (active_count - CAPACITY_SOFT_LIMIT) / float(CAPACITY_SOFT_LIMIT)
    return max(0.5, 1.0 - min(0.5, over))


def evaluate_admission(
    fact_text: str,
    candidates: Optional[Sequence[Any]] = None,
    *,
    source_trust: float = 0.8,
    provenance_kind: str = "explicit",
    active_count: Optional[int] = None,
    force: bool = False,
) -> AdmissionDecision:
    """Score a candidate write and decide STORE / MERGE / DROP."""
    cands = list(candidates or [])
    novelty = _novelty(cands)
    importance = _importance(fact_text)
    credibility = _credibility(source_trust, provenance_kind)

    score = (0.40 * novelty + 0.35 * importance + 0.25 * credibility) * _headroom(active_count)

    if force:
        return AdmissionDecision("STORE", score, novelty, importance, credibility, "forced")

    if not fact_text.strip():
        return AdmissionDecision("DROP", 0.0, novelty, importance, credibility, "empty")

    # Hard gate: pure low-value / tautology content is never stored (E32).
    if importance <= 0.05 and not cands:
        return AdmissionDecision("DROP", round(score, 4), novelty, importance, credibility,
                                 "low-value/tautology content")

    if score >= STORE_THRESHOLD:
        action, reason = "STORE", "score above store threshold"
    elif score >= MERGE_THRESHOLD and cands:
        action, reason = "MERGE", "mid-band score with existing near-duplicate"
    else:
        action, reason = "DROP", "low novelty/importance/credibility"

    if action == "DROP":
        log_atomic(f"E32 admission DROP (score={score:.2f}): '{fact_text[:60]}'")
    return AdmissionDecision(action, round(score, 4), novelty, importance, credibility, reason)
