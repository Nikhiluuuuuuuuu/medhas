"""E37/E35/E5 — Interference matrix, working-memory eviction, recognition filter.

E37 Interference: quantify proactive (old blocks new) and retroactive (new corrupts
    old) interference between similar memories, and resolve by disambiguation.
E35 WM eviction: bounded working memory with an explicit eviction policy
    (importance × recency × relevance), evicting to archival rather than dropping.
E5  Recognition-before-recall: a cheap novelty gate that answers "have I seen this
    before?" before paying for full hybrid retrieval.
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from medhas.storage import DatabasePool
from medhas.utils import log_atomic, log_error, measure_latency

# ------------------------------------------------------------------ E37

INTERFERENCE_SIM_FLOOR = 0.82   # above this, two memories genuinely compete


async def interference_matrix(user_id: str, limit: int = 60) -> List[Dict[str, Any]]:
    """Pairs of active memories similar enough to interfere, with direction + severity.

    proactive  = older memory suppresses retrieval of the newer one
    retroactive = newer memory suppresses/corrupts the older one
    """
    async with measure_latency("agi.interference.interference_matrix"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT a.id AS a_id, a.fact_text AS a_text, a.created_at AS a_at,
                           a.importance_score AS a_imp,
                           b.id AS b_id, b.fact_text AS b_text, b.created_at AS b_at,
                           b.importance_score AS b_imp,
                           1 - (a.embedding <=> b.embedding) AS sim
                    FROM atomic_facts a
                    JOIN atomic_facts b
                      ON a.user_id = b.user_id AND a.id < b.id
                    WHERE a.user_id = $1 AND a.is_active AND b.is_active
                      AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
                      AND 1 - (a.embedding <=> b.embedding) >= $2
                    ORDER BY sim DESC
                    LIMIT $3;
                    """,
                    user_id, INTERFERENCE_SIM_FLOOR, limit,
                )
            out: List[Dict[str, Any]] = []
            for r in rows:
                older_first = r["a_at"] <= r["b_at"]
                older_imp = float(r["a_imp"] if older_first else r["b_imp"])
                newer_imp = float(r["b_imp"] if older_first else r["a_imp"])
                direction = "proactive" if older_imp >= newer_imp else "retroactive"
                out.append({
                    "a_id": r["a_id"], "a_text": r["a_text"],
                    "b_id": r["b_id"], "b_text": r["b_text"],
                    "similarity": round(float(r["sim"]), 4),
                    "direction": direction,
                    "severity": round(float(r["sim"]) * (1.0 + abs(older_imp - newer_imp) / 10.0), 4),
                })
            return out
        except Exception as e:
            log_error(f"interference_matrix failed: {e}")
            return []


async def resolve_interference(user_id: str, apply: bool = False) -> Dict[str, Any]:
    """Disambiguate interfering pairs by annotating context, not by deleting either side."""
    pairs = await interference_matrix(user_id)
    resolved = 0
    try:
        if apply and pairs:
            async with DatabasePool.acquire() as conn:
                for p in pairs[:20]:
                    older, newer = (p["a_id"], p["b_id"]) if p["direction"] == "proactive" else (p["b_id"], p["a_id"])
                    await conn.execute(
                        """
                        UPDATE atomic_facts
                        SET note = COALESCE(note,'') || ' [interferes_with:' || $2::text || ']',
                            linked_ids = (
                                SELECT ARRAY(SELECT DISTINCT unnest(COALESCE(linked_ids, ARRAY[]::uuid[]) || ARRAY[$2]::uuid[]))
                            )
                        WHERE id = $1 AND position(('[interferes_with:' || $2::text) in COALESCE(note,'')) = 0;
                        """,
                        newer, older,
                    )
                    resolved += 1
            log_atomic(f"E37 annotated {resolved} interfering memory pairs")
    except Exception as e:
        log_error(f"resolve_interference failed: {e}")
    return {"pairs": len(pairs), "resolved": resolved, "detail": pairs[:10]}


# ------------------------------------------------------------------ E35

WM_MAX_BLOCK_CHARS = 4000
WM_MAX_TOTAL_CHARS = 12000


def eviction_scores(blocks: Dict[str, str], query: str = "") -> List[Tuple[str, float]]:
    """Rank working-memory blocks by keep-value (importance × recency × relevance)."""
    q_tokens = {t for t in (query or "").lower().split() if len(t) > 2}
    priority = {"persona": 1.0, "user_profile": 0.9, "active_goals": 0.8, "scratchpad": 0.5}
    scored: List[Tuple[str, float]] = []
    for name, content in (blocks or {}).items():
        text = str(content or "")
        base = priority.get(name, 0.6)
        relevance = 0.0
        if q_tokens and text:
            lc = text.lower()
            relevance = sum(1 for t in q_tokens if t in lc) / max(1, len(q_tokens))
        density = min(1.0, len(text) / 800.0)
        scored.append((name, round(base * (0.6 + 0.4 * relevance) * (0.7 + 0.3 * density), 4)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


async def evict_working_memory(
    user_id: str,
    query: str = "",
    max_total_chars: int = WM_MAX_TOTAL_CHARS,
) -> Dict[str, Any]:
    """Enforce a working-memory budget, archiving evicted content instead of losing it."""
    from medhas.memory.archival import archive_memory
    from medhas.memory.working import get_blocks, update_block

    async with measure_latency("agi.interference.evict_working_memory"):
        try:
            record = await get_blocks(user_id)
            block_map = record.blocks.to_block_map()
            blocks: Dict[str, str] = {label: b.value for label, b in block_map.items()}
            read_only = {label for label, b in block_map.items() if b.read_only}
            total = sum(len(v or "") for v in blocks.values())
            if total <= max_total_chars:
                return {"evicted": 0, "total_chars": total, "status": "within_budget"}

            ranked = eviction_scores(blocks, query)
            evicted = 0
            for name, _score in reversed(ranked):     # lowest keep-value first
                if total <= max_total_chars:
                    break
                content = str(blocks.get(name) or "")
                if not content or name == "persona" or name in read_only:
                    continue
                await archive_memory(user_id, f"[WM:{name}] {content}")
                keep = content[:600]
                await update_block(user_id, name, keep + ("…" if len(content) > 600 else ""))
                total -= max(0, len(content) - len(keep))
                evicted += 1
            log_atomic(f"E35 evicted {evicted} working-memory blocks to archival")
            return {"evicted": evicted, "total_chars": total, "status": "evicted"}
        except Exception as e:
            log_error(f"evict_working_memory failed: {e}")
            return {"evicted": 0, "error": str(e)}


# ------------------------------------------------------------------- E5

async def recognize(user_id: str, text: str) -> Dict[str, Any]:
    """Recognition-before-recall gate: cheap 'have I seen this?' check.

    Uses the exact content hash (O(1) index hit) before any embedding/hybrid search,
    so repeat inputs short-circuit the expensive retrieval path.
    """
    from medhas.memory.atomic.hashing import content_hash

    h = content_hash(text or "")
    try:
        async with DatabasePool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, fact_text, created_at FROM atomic_facts
                WHERE user_id = $1 AND content_hash = $2 AND is_active = TRUE
                LIMIT 1;
                """,
                user_id, h,
            )
        if row:
            return {"known": True, "novel": False, "fact_id": row["id"],
                    "fact_text": row["fact_text"], "method": "content_hash"}
        return {"known": False, "novel": True, "fact_id": None, "method": "content_hash"}
    except Exception as e:
        log_error(f"recognize failed: {e}")
        return {"known": False, "novel": True, "error": str(e)}
