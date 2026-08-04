"""E16/E17/E18 — Metacognitive retrieval routing, self-improvement critic, abstention.

E16 Routing: choose the cheapest retrieval strategy that can answer the query
    (recognition → vector-only → hybrid → graph multi-hop → full consolidation read)
    instead of always paying for the full pipeline.
E17 Self-improvement: score retrieval quality after each turn and adjust weights,
    logging the outcome so the system provably improves over time.
E18 Abstention: refuse to answer from memory when evidence is weak (calibrated by
    the E29 feeling-of-knowing signal) rather than hallucinating.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from infrastructure.db import DatabasePool
from utils import log_atomic, log_error, log_working, measure_latency

# ------------------------------------------------------------------- E16

STRATEGIES = ("recognition", "vector", "hybrid", "graph", "deep")

_MULTIHOP_MARKERS = (
    "who else", "related to", "connected", "because", "why did", "how did",
    "compare", "between", "chain", "lead to", "caused", "impact of", "through",
)
_TEMPORAL_MARKERS = ("when", "before", "after", "used to", "still", "changed", "last time", "history")
_SIMPLE_MARKERS = ("what is my", "what's my", "my name", "do i", "remind me", "recall")


@dataclass
class RoutingDecision:
    strategy: str
    reason: str
    use_graph: bool = False
    use_temporal: bool = False
    top_k: int = 5
    signals: Dict[str, Any] = field(default_factory=dict)


def route_query(query: str, *, recognized: bool = False, complexity_hint: Optional[str] = None) -> RoutingDecision:
    """Pick the cheapest sufficient retrieval strategy for this query (E16)."""
    q = (query or "").lower().strip()
    words = len(q.split())

    if recognized:
        return RoutingDecision("recognition", "exact prior content recognized", top_k=1)

    multihop = any(m in q for m in _MULTIHOP_MARKERS)
    temporal = any(m in q for m in _TEMPORAL_MARKERS)
    simple = any(m in q for m in _SIMPLE_MARKERS)

    if complexity_hint == "deep" or (multihop and temporal):
        return RoutingDecision("deep", "multi-hop + temporal reasoning required",
                               use_graph=True, use_temporal=True, top_k=10,
                               signals={"multihop": multihop, "temporal": temporal})
    if multihop:
        return RoutingDecision("graph", "relational/multi-hop query", use_graph=True, top_k=8,
                               signals={"multihop": True})
    if temporal:
        return RoutingDecision("hybrid", "temporal query needs validity filtering",
                               use_temporal=True, top_k=8, signals={"temporal": True})
    if simple and words <= 8:
        return RoutingDecision("vector", "short direct factual lookup", top_k=3)
    return RoutingDecision("hybrid", "default hybrid retrieval", top_k=5)


# ------------------------------------------------------------------- E18

ABSTAIN_FLOOR = 0.35


def should_abstain(results: Sequence[Any], fok: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Calibrated abstention decision (E18) built on the E29 FoK signal."""
    from agi.metamemory import feeling_of_knowing

    signal = fok or feeling_of_knowing(results)
    abstain = bool(signal.get("abstain", True)) or float(signal.get("fok", 0.0)) < ABSTAIN_FLOOR
    return {
        "abstain": abstain,
        "confidence": signal.get("fok", 0.0),
        "evidence": signal.get("evidence", 0),
        "message": (
            "I don't have that in memory — I'd be guessing."
            if abstain else ""
        ),
    }


# ------------------------------------------------------------------- E17

@dataclass
class RetrievalOutcome:
    query: str
    strategy: str
    results: int
    top_similarity: float
    used_in_answer: int = 0
    latency_ms: float = 0.0


class SelfImprovementCritic:
    """Scores retrieval quality per turn and adapts routing weights (E17)."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []
        self.strategy_stats: Dict[str, Dict[str, float]] = {s: {"n": 0.0, "score": 0.0} for s in STRATEGIES}

    def score(self, outcome: RetrievalOutcome) -> float:
        """0..1 quality score: did retrieval return usable, high-similarity evidence fast?"""
        if outcome.results == 0:
            quality = 0.0
        else:
            hit = min(1.0, outcome.top_similarity / 0.85)
            usage = min(1.0, outcome.used_in_answer / max(1, min(3, outcome.results)))
            speed = 1.0 if outcome.latency_ms <= 150 else max(0.3, 150.0 / outcome.latency_ms)
            quality = 0.5 * hit + 0.3 * usage + 0.2 * speed
        st = self.strategy_stats.setdefault(outcome.strategy, {"n": 0.0, "score": 0.0})
        st["n"] += 1
        st["score"] += quality
        self.history.append({"query": outcome.query[:120], "strategy": outcome.strategy,
                             "quality": round(quality, 4), "results": outcome.results})
        return quality

    def best_strategy(self) -> Optional[str]:
        ranked = [(s, v["score"] / v["n"]) for s, v in self.strategy_stats.items() if v["n"] >= 3]
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[0][0]

    def report(self) -> Dict[str, Any]:
        return {
            "turns": len(self.history),
            "mean_quality": round(
                sum(h["quality"] for h in self.history) / len(self.history), 4
            ) if self.history else 0.0,
            "by_strategy": {
                s: round(v["score"] / v["n"], 4) for s, v in self.strategy_stats.items() if v["n"]
            },
            "best_strategy": self.best_strategy(),
        }

    async def persist(self, user_id: str) -> None:
        """Store the critic's rolling report for longitudinal improvement tracking."""
        try:
            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO eval_runs (user_id, suite, metrics) VALUES ($1,'self_improvement',$2::jsonb);",
                    user_id, json.dumps(self.report()),
                )
        except Exception as e:
            log_error(f"critic persist failed: {e}")


#: process-wide critic
critic = SelfImprovementCritic()


async def log_memory_event(user_id: Optional[str], event: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """E24 — structured observability event."""
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_events (user_id, event, payload) VALUES ($1,$2,$3::jsonb);",
                user_id, event, json.dumps(payload or {}),
            )
    except Exception as e:
        log_error(f"log_memory_event failed: {e}")
