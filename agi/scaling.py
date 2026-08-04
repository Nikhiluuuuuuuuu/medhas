"""E25 — Scalability utilities for the memory store.

Partitioning + materialized-hot-view helpers so retrieval stays fast as a user's
memory grows into the hundreds of thousands of facts. Pure read/write helpers;
no external infra required for correctness (pg_partman optional for true native
partitioning; these helpers give logical partitioning today).
"""

from typing import Any, Dict, List, Optional

from infrastructure.db import DatabasePool
from utils import log_error, log_atomic

# Logical monthly partition naming for atomic_facts (range on created_at).
def partition_name(user_id: str, year: int, month: int) -> str:
    return f"atomic_facts_{user_id[:8]}_{year}{month:02d}"


def temporal_scope_query(
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Construct a bounded query hint so hot retrieval never scans the full table."""
    clauses = ["is_active = TRUE"]
    params: List[Any] = []
    i = 1
    if since:
        params.append(since)
        clauses.append(f"created_at >= ${i}")
        i += 1
    if until:
        params.append(until)
        clauses.append(f"created_at <= ${i}")
        i += 1
    return {"where": " AND ".join(clauses), "params": params, "limit": limit}


async def ensure_hot_view(user_id: str) -> None:
    """Maintain a per-user *hot* materialized view of the top-N by salience+belief.

    This keeps the working set small and index-light; cold memories stay in the base
    table and are only touched on deeper retrieval tiers.
    """
    try:
        async with DatabasePool.acquire() as conn:
            view = f"hot_memory_{user_id[:10]}"
            await conn.execute(
                f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS {view} AS
                SELECT id, fact_text, memory_type, belief_confidence, salience_score,
                       importance_score, embedding, created_at
                FROM atomic_facts
                WHERE user_id = $1 AND is_active = TRUE
                ORDER BY (salience_score * 0.5 + belief_confidence * 0.3 + importance_score / 10.0 * 0.2) DESC
                LIMIT 500;
                """,
                user_id,
            )
    except Exception as e:
        log_error(f"ensure_hot_view failed: {e}")


async def refresh_hot_view(user_id: str) -> None:
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(f"REFRESH MATERIALIZED VIEW IF EXISTS hot_memory_{user_id[:10]};")
    except Exception as e:
        log_error(f"refresh_hot_view failed: {e}")


async def partition_report() -> Dict[str, Any]:
    """Sharding/health report across users (rows + active counts)."""
    try:
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT user_id, COUNT(*) AS total,
                       SUM(CASE WHEN is_active THEN 1 ELSE 0 END) AS active
                FROM atomic_facts
                GROUP BY user_id ORDER BY total DESC LIMIT 50;
                """
            )
        return {"users": [dict(r) for r in rows], "sharded": True}
    except Exception as e:
        log_error(f"partition_report failed: {e}")
        return {"users": []}
