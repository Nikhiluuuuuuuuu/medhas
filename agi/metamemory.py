"""E29 — Meta-memory: knowing what you know (and what you don't).

Feeling-of-knowing / known-unknowns. Powers calibrated abstention (E18): the agent
can say "I don't have that stored" instead of hallucinating a plausible memory.
"""

import re
from typing import Any, Dict, List, Optional, Sequence

from infrastructure.db import DatabasePool
from utils import log_atomic, log_error, measure_latency

_STOP = {
    "what", "when", "where", "who", "why", "how", "the", "a", "an", "is", "are",
    "do", "does", "did", "my", "me", "i", "you", "your", "of", "to", "in", "on",
    "for", "about", "tell", "know", "remember", "was", "were", "and", "it",
}

#: below this aggregate evidence the system should abstain rather than answer
FOK_ABSTAIN_THRESHOLD = 0.45
#: a hit only "supports" an answer if its cosine sim clears this AND it shares entity tokens
STRONG_SIM = 0.55
#: minimum fraction of query entity tokens that must appear in the top hit
ENTITY_OVERLAP_MIN = 0.25


def topic_of(query: str) -> str:
    toks = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if t not in _STOP and len(t) > 2]
    return " ".join(toks[:4]) if toks else (query or "").strip().lower()[:60]


def _entity_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if len(t) > 2 and t not in _STOP}


def _overlap(query: str, fact_text: str) -> float:
    q = _entity_tokens(query)
    if not q:
        return 0.0
    f = _entity_tokens(fact_text)
    return len(q & f) / len(q)


def feeling_of_knowing(results: Sequence[Any], query: str = "") -> Dict[str, Any]:
    """Compute a calibrated FoK signal from retrieval results.

    Combines top similarity, evidence count and mean belief — BUT a hit only counts
    as supporting evidence when it both clears a strong-similarity bar AND shares
    entity tokens with the query (E18 calibrated gate). This prevents a loosely
    similar but topically unrelated fact from suppressing abstention.
    """
    if not results:
        return {
            "fok": 0.0, "abstain": True, "evidence": 0, "top_similarity": 0.0,
            "mean_belief": 0.0, "reason": "no matching memories stored",
        }
    best_sim = 0.0
    best_overlap = 0.0
    for r in results:
        sim = float(getattr(r, "similarity", 0.0) or 0.0)
        ft = getattr(r, "fact_text", "")
        ov = _overlap(query, ft) if query else 0.0
        if sim > best_sim:
            best_sim = sim
            best_overlap = ov
        elif sim == best_sim and ov > best_overlap:
            best_overlap = ov
    beliefs = [float(getattr(r, "belief_confidence", 0.7) or 0.7) for r in results]
    mean_belief = sum(beliefs) / len(beliefs)
    coverage = min(1.0, len(results) / 3.0)

    # Supporting evidence requires BOTH a strong similarity and entity overlap.
    supported = (best_sim >= STRONG_SIM) and (best_overlap >= ENTITY_OVERLAP_MIN)
    if not supported:
        # No genuinely relevant hit -> low confidence -> abstain.
        fok = 0.15 + 0.20 * coverage
        return {
            "fok": round(fok, 4), "abstain": True, "evidence": len(results),
            "top_similarity": round(best_sim, 4), "mean_belief": round(mean_belief, 4),
            "best_overlap": round(best_overlap, 4),
            "reason": "top hit unrelated to query (low similarity / no entity overlap)",
        }

    fok = 0.55 * best_sim + 0.25 * mean_belief + 0.20 * coverage
    abstain = fok < FOK_ABSTAIN_THRESHOLD
    return {
        "fok": round(fok, 4),
        "abstain": abstain,
        "evidence": len(results),
        "top_similarity": round(best_sim, 4),
        "mean_belief": round(mean_belief, 4),
        "best_overlap": round(best_overlap, 4),
        "reason": "insufficient stored evidence" if abstain else "sufficient stored evidence",
    }


async def record_topic_knowledge(
    user_id: str,
    topic: str,
    known_count: int,
    mean_confidence: float,
) -> None:
    """Persist what the system knows about a topic (upsert)."""
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO meta_memory (user_id, topic, known_count, mean_confidence,
                                         is_known_unknown, updated_at)
                VALUES ($1,$2,$3,$4,$5,CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, topic) DO UPDATE SET
                    known_count = EXCLUDED.known_count,
                    mean_confidence = EXCLUDED.mean_confidence,
                    is_known_unknown = EXCLUDED.is_known_unknown,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                user_id, topic, int(known_count), float(mean_confidence), known_count == 0,
            )
    except Exception as e:
        log_error(f"record_topic_knowledge failed: {e}")


async def assess(user_id: str, query: str, results: Sequence[Any]) -> Dict[str, Any]:
    """Full meta-memory assessment for a query: FoK + persist topic knowledge."""
    async with measure_latency("agi.metamemory.assess"):
        fok = feeling_of_knowing(results, query)
        topic = topic_of(query)
        if topic:
            await record_topic_knowledge(
                user_id, topic, fok["evidence"], fok["mean_belief"]
            )
        if fok["abstain"]:
            log_atomic(f"E29 known-unknown registered for topic '{topic}' (fok={fok['fok']})")
        return {**fok, "topic": topic}


async def known_unknowns(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Topics the system has been asked about but has no memory for."""
    try:
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT topic, known_count, mean_confidence, updated_at
                FROM meta_memory
                WHERE user_id = $1 AND is_known_unknown = TRUE
                ORDER BY updated_at DESC LIMIT $2;
                """,
                user_id, limit,
            )
            return [dict(r) for r in rows]
    except Exception as e:
        log_error(f"known_unknowns failed: {e}")
        return []


async def knowledge_map(user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """What the agent believes it knows, ranked by confidence (introspection surface)."""
    try:
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT topic, known_count, mean_confidence, is_known_unknown, updated_at
                FROM meta_memory WHERE user_id=$1
                ORDER BY mean_confidence DESC, known_count DESC LIMIT $2;
                """,
                user_id, limit,
            )
            return [dict(r) for r in rows]
    except Exception as e:
        log_error(f"knowledge_map failed: {e}")
        return []
