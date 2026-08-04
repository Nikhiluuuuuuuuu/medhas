"""E13/E14/E15/E28/E33 — Forgetting, salience, reconsolidation, affect, protected core.

E13 Algorithmic forgetting: Ebbinghaus decay with a per-fact half-life; low-salience
    stale memories deactivate, high-belief/protected ones persist.
E14 Salience learning: importance grows with access frequency/recency/contradiction.
E15 Reconsolidation: recall makes a memory labile — access metadata updates and a
    contradiction during recall triggers evolution instead of a silent duplicate.
E28 Affective weighting: high arousal flattens the decay curve (flashbulb effect).
E33 Protected core: high-belief memories are frozen against decay/overwrite.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from medhas.storage import DatabasePool
from medhas.utils import log_atomic, log_error, measure_latency

PROTECT_BELIEF = 0.90
PROTECT_IMPORTANCE = 9.0
FORGET_RETENTION_FLOOR = 0.05
FORGET_MIN_AGE_DAYS = 7.0


# ------------------------------------------------------------ decay / retention

def effective_half_life(
    half_life_days: float,
    arousal: float = 0.0,
    importance: float = 5.0,
    access_count: int = 0,
) -> float:
    """Half-life stretched by emotional arousal (E28), importance and rehearsal (E14).

    arousal in 0..1 can up to triple the half-life (flashbulb memories decay flatter).
    """
    hl = max(0.1, float(half_life_days))
    hl *= 1.0 + 2.0 * max(0.0, min(1.0, float(arousal)))          # E28
    hl *= 1.0 + 0.10 * max(0.0, min(10.0, float(importance)))     # E14
    hl *= 1.0 + 0.15 * math.log1p(max(0, int(access_count)))      # spacing/testing effect
    return hl


def retention(
    created_at: datetime,
    *,
    last_accessed_at: Optional[datetime] = None,
    half_life_days: float = 7.0,
    arousal: float = 0.0,
    importance: float = 5.0,
    access_count: int = 0,
) -> float:
    """Ebbinghaus retention R = exp(-t / HL), t measured from last reactivation."""
    anchor = last_accessed_at or created_at
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    elapsed_days = max(0.0, (datetime.now(timezone.utc) - anchor).total_seconds() / 86400.0)
    hl = effective_half_life(half_life_days, arousal, importance, access_count)
    return max(0.001, min(1.0, math.exp(-elapsed_days / hl)))


# ---------------------------------------------------- reconsolidation (E15/E14)

async def reconsolidate(fact_ids: Sequence[UUID]) -> int:
    """Mark recalled memories as reactivated: bump access_count, last_accessed_at,
    salience and importance (retrieval practice strengthens — the testing effect)."""
    ids = [f for f in fact_ids if f]
    if not ids:
        return 0
    async with measure_latency("agi.forgetting.reconsolidate"):
        try:
            async with DatabasePool.acquire() as conn:
                status = await conn.execute(
                    """
                    UPDATE atomic_facts
                    SET access_count = access_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP,
                        salience_score = LEAST(1.0, salience_score + 0.05),
                        importance_score = LEAST(10.0, importance_score + 0.25)
                    WHERE id = ANY($1::uuid[]);
                    """,
                    list(ids),
                )
            n = int(status.split()[-1]) if status else 0
            return n
        except Exception as e:
            log_error(f"reconsolidate failed: {e}")
            return 0


async def set_affect(fact_id: UUID, valence: float, arousal: float) -> None:
    """E28 — attach emotional valence/arousal to a memory."""
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                """
                UPDATE atomic_facts
                SET affect_valence = $2::double precision,
                    affect_arousal = $3::double precision,
                    decay_half_life_days = GREATEST(
                        decay_half_life_days,
                        (1.0::double precision + 2.0::double precision * $3::double precision) * 7.0::double precision
                    )
                WHERE id = $1;
                """,
                fact_id, max(-1.0, min(1.0, valence)), max(0.0, min(1.0, arousal)),
            )
    except Exception as e:
        log_error(f"set_affect failed: {e}")


# ------------------------------------------------------- protected core (E33)

async def protect_core_memories(user_id: str) -> int:
    """Freeze high-belief / high-importance memories against decay and overwrite."""
    try:
        async with DatabasePool.acquire() as conn:
            status = await conn.execute(
                """
                UPDATE atomic_facts
                SET is_protected = TRUE
                WHERE user_id = $1 AND is_active = TRUE AND is_protected = FALSE
                  AND (belief_confidence >= $2 OR importance_score >= $3);
                """,
                user_id, PROTECT_BELIEF, PROTECT_IMPORTANCE,
            )
        return int(status.split()[-1]) if status else 0
    except Exception as e:
        log_error(f"protect_core_memories failed: {e}")
        return 0


async def is_protected(fact_id: UUID) -> bool:
    try:
        async with DatabasePool.acquire() as conn:
            v = await conn.fetchval("SELECT is_protected FROM atomic_facts WHERE id=$1;", fact_id)
            return bool(v)
    except Exception:
        return False


# ------------------------------------------------------- forgetting sweep (E13)

async def run_forgetting_sweep(user_id: str, dry_run: bool = False) -> Dict[str, Any]:
    """Deactivate stale, low-salience, unprotected memories. Protected/high-belief persist."""
    async with measure_latency("agi.forgetting.run_forgetting_sweep"):
        forgotten: List[str] = []
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, fact_text, created_at, last_accessed_at, decay_half_life_days,
                           affect_arousal, importance_score, access_count, belief_confidence,
                           is_protected
                    FROM atomic_facts
                    WHERE user_id = $1 AND is_active = TRUE
                      AND created_at < NOW() - ($2 || ' days')::interval;
                    """,
                    user_id, str(FORGET_MIN_AGE_DAYS),
                )
                for r in rows:
                    if r["is_protected"] or float(r["belief_confidence"]) >= PROTECT_BELIEF:
                        continue
                    if str(r["fact_text"]).startswith(("[Reflection]", "[Pattern]", "[Gist]")):
                        continue
                    ret = retention(
                        r["created_at"],
                        last_accessed_at=r["last_accessed_at"],
                        half_life_days=float(r["decay_half_life_days"]),
                        arousal=float(r["affect_arousal"]),
                        importance=float(r["importance_score"]),
                        access_count=int(r["access_count"]),
                    )
                    if ret < FORGET_RETENTION_FLOOR:
                        forgotten.append(str(r["id"]))
                        if not dry_run:
                            await conn.execute(
                                """
                                UPDATE atomic_facts
                                SET is_active = FALSE, expired_at = CURRENT_TIMESTAMP,
                                    valid_to = COALESCE(valid_to, CURRENT_TIMESTAMP)
                                WHERE id = $1;
                                """,
                                r["id"],
                            )
            if forgotten:
                log_atomic(f"E13 forgetting sweep deactivated {len(forgotten)} stale memories")
            return {"forgotten": len(forgotten), "ids": forgotten, "dry_run": dry_run}
        except Exception as e:
            log_error(f"run_forgetting_sweep failed: {e}")
            return {"forgotten": 0, "ids": [], "error": str(e)}


# ------------------------------------------------- spaced reinforcement (E9)

async def schedule_review(fact_id: UUID, success: bool = True) -> Optional[datetime]:
    """Leitner-style spaced repetition: successful reactivation lengthens the interval."""
    try:
        async with DatabasePool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT review_interval_days, review_count FROM atomic_facts WHERE id=$1;", fact_id
            )
            if not row:
                return None
            interval = float(row["review_interval_days"] or 1.0)
            interval = min(180.0, interval * 2.0) if success else max(0.5, interval / 2.0)
            next_at = datetime.now(timezone.utc) + timedelta(days=interval)
            await conn.execute(
                """
                UPDATE atomic_facts
                SET review_interval_days = $2, next_review_at = $3,
                    review_count = review_count + 1
                WHERE id = $1;
                """,
                fact_id, interval, next_at,
            )
            return next_at
    except Exception as e:
        log_error(f"schedule_review failed: {e}")
        return None


async def due_for_review(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Facts whose spaced-repetition review is due (or never scheduled but important)."""
    try:
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, fact_text, importance_score, next_review_at
                FROM atomic_facts
                WHERE user_id = $1 AND is_active = TRUE
                  AND (next_review_at IS NULL OR next_review_at <= CURRENT_TIMESTAMP)
                ORDER BY importance_score DESC, created_at ASC
                LIMIT $2;
                """,
                user_id, limit,
            )
            return [dict(r) for r in rows]
    except Exception as e:
        log_error(f"due_for_review failed: {e}")
        return []
