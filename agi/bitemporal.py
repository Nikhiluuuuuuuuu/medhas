"""E10/E11/E12 — Bitemporal contradiction lattice, Bayesian belief, provenance.

E10: facts carry (created_at, valid_from, valid_to) so "what was true at T" is a
     first-class query and contradictory facts coexist instead of overwriting.
E11: belief_confidence updated with an incremental odds-form Bayesian posterior.
E12: retrieval payload carries source episode, belief, contradicted_by, validity.

References: TOKI bitemporal operator algebra (2606.06240); Zep/Graphiti
created_at/valid_at/invalid_at; existing memory/graph/belief_revision.py.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from infrastructure.db import DatabasePool
from utils import log_atomic, log_error, measure_latency

MIN_BELIEF = 0.01
MAX_BELIEF = 0.99


# ---------------------------------------------------------------- belief (E11)

def _odds(p: float) -> float:
    p = max(MIN_BELIEF, min(MAX_BELIEF, p))
    return p / (1.0 - p)


def _prob(o: float) -> float:
    return max(MIN_BELIEF, min(MAX_BELIEF, o / (1.0 + o)))


def bayesian_update(prior: float, likelihood: float, supports: bool = True) -> float:
    """Incremental odds-form posterior.

    supports=True  -> corroboration multiplies odds by LR = l/(1-l)
    supports=False -> contradiction divides odds by the same LR
    """
    lr = _odds(likelihood)
    o = _odds(prior)
    return _prob(o * lr if supports else o / lr)


async def revise_fact_belief(
    fact_id: UUID,
    likelihood: float = 0.75,
    supports: bool = True,
) -> float:
    """Apply a Bayesian belief revision to one atomic fact. Returns new belief."""
    async with measure_latency("agi.bitemporal.revise_fact_belief"):
        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT belief_confidence FROM atomic_facts WHERE id = $1;", fact_id
                )
                if not row:
                    return 0.0
                new_belief = bayesian_update(float(row["belief_confidence"]), likelihood, supports)
                await conn.execute(
                    "UPDATE atomic_facts SET belief_confidence = $2 WHERE id = $1;",
                    fact_id, new_belief,
                )
                return new_belief
        except Exception as e:
            log_error(f"revise_fact_belief failed: {e}")
            return 0.0


# ------------------------------------------------------ bitemporal lattice (E10)

async def invalidate_fact(
    fact_id: UUID,
    *,
    invalidated_by: Optional[UUID] = None,
    valid_to: Optional[datetime] = None,
) -> None:
    """Close a fact's valid-time interval without deleting it (contradictions coexist)."""
    vt = valid_to or datetime.now(timezone.utc)
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                """
                UPDATE atomic_facts
                SET valid_to = COALESCE(valid_to, $2), invalidated_by = COALESCE($3, invalidated_by)
                WHERE id = $1;
                """,
                fact_id, vt, invalidated_by,
            )
            if invalidated_by is not None:
                await conn.execute(
                    """
                    UPDATE atomic_facts
                    SET contradicted_by = array_append(COALESCE(contradicted_by, ARRAY[]::uuid[]), $2)
                    WHERE id = $1 AND NOT ($2 = ANY(COALESCE(contradicted_by, ARRAY[]::uuid[])));
                    """,
                    fact_id, invalidated_by,
                )
        log_atomic(f"E10 invalidated fact {fact_id} at {vt.isoformat()}")
    except Exception as e:
        log_error(f"invalidate_fact failed: {e}")


async def mark_contradiction(fact_id: UUID, contradicting_id: UUID) -> None:
    """Record a contradiction pointer both ways without closing validity (E12)."""
    try:
        async with DatabasePool.acquire() as conn:
            for a, b in ((fact_id, contradicting_id), (contradicting_id, fact_id)):
                await conn.execute(
                    """
                    UPDATE atomic_facts
                    SET contradicted_by = array_append(COALESCE(contradicted_by, ARRAY[]::uuid[]), $2)
                    WHERE id = $1 AND NOT ($2 = ANY(COALESCE(contradicted_by, ARRAY[]::uuid[])));
                    """,
                    a, b,
                )
    except Exception as e:
        log_error(f"mark_contradiction failed: {e}")


async def facts_valid_at(
    user_id: str,
    at: datetime,
    limit: int = 50,
    memory_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """E10/E20 — 'what was true at time T' snapshot query over the valid-time lattice."""
    async with measure_latency("agi.bitemporal.facts_valid_at"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, fact_text, memory_type, belief_confidence,
                           valid_from, valid_to, created_at, source_episode_id, contradicted_by
                    FROM atomic_facts
                    WHERE user_id = $1
                      AND valid_from <= $2
                      AND (valid_to IS NULL OR valid_to > $2)
                      AND ($4::text IS NULL OR memory_type = $4)
                    ORDER BY belief_confidence DESC, valid_from DESC
                    LIMIT $3;
                    """,
                    user_id, at, limit, memory_type,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            log_error(f"facts_valid_at failed: {e}")
            return []


async def fact_provenance(fact_id: UUID) -> Dict[str, Any]:
    """E12 — full provenance + uncertainty payload for one fact."""
    try:
        async with DatabasePool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT f.id, f.fact_text, f.belief_confidence, f.source_trust,
                       f.provenance_kind, f.source_episode_id, f.contradicted_by,
                       f.valid_from, f.valid_to, f.created_at, f.is_quarantined,
                       e.content AS episode_content, e.reference_time AS episode_time
                FROM atomic_facts f
                LEFT JOIN episodes e ON e.id = f.source_episode_id
                WHERE f.id = $1;
                """,
                fact_id,
            )
            return dict(row) if row else {}
    except Exception as e:
        log_error(f"fact_provenance failed: {e}")
        return {}
