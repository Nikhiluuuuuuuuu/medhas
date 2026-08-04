"""E27 — Prospective memory: future intentions that fire on a cue or a time.

Reference: PM-Bench (2607.12385). Distinct from episodic (past) memory: an intention
is carried through unrelated activity and must fire exactly when its cue occurs,
and NOT otherwise.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from infrastructure.db import DatabasePool
from utils import log_atomic, log_error, measure_latency

_STOPWORDS = {
    "the", "a", "an", "when", "if", "i", "me", "my", "to", "of", "at", "on", "in",
    "and", "is", "are", "do", "does", "get", "got", "for", "with", "that", "this",
}


def _cue_tokens(cue: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (cue or "").lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 2]


async def add_intention(
    user_id: str,
    intent: str,
    *,
    cue_text: Optional[str] = None,
    trigger_at: Optional[datetime] = None,
    agent_id: Optional[str] = None,
) -> UUID:
    """Store a future intention. Provide cue_text (event cue) and/or trigger_at (time cue)."""
    cue_kind = "time" if trigger_at is not None and not cue_text else "event"
    async with measure_latency("agi.prospective.add_intention"):
        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO prospective_memory
                        (user_id, agent_id, intent, cue_text, cue_kind, trigger_at)
                    VALUES ($1,$2,$3,$4,$5,$6)
                    RETURNING id;
                    """,
                    user_id, agent_id, intent, cue_text, cue_kind, trigger_at,
                )
            log_atomic(f"E27 intention stored: '{intent}' (cue={cue_text or trigger_at})")
            return row["id"]
        except Exception as e:
            log_error(f"add_intention failed: {e}")
            raise


async def check_cues(
    user_id: str,
    current_context: str = "",
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Return intentions whose cue fires RIGHT NOW (event match or time reached).

    Event cue fires only when the majority of its content tokens appear in the
    current context — prevents spurious firing on unrelated turns.
    """
    ts = now or datetime.now(timezone.utc)
    ctx = (current_context or "").lower()
    fired: List[Dict[str, Any]] = []
    async with measure_latency("agi.prospective.check_cues"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, intent, cue_text, cue_kind, trigger_at
                    FROM prospective_memory
                    WHERE user_id = $1 AND is_done = FALSE AND fired_at IS NULL;
                    """,
                    user_id,
                )
                for r in rows:
                    hit = False
                    if r["trigger_at"] is not None and r["trigger_at"] <= ts:
                        hit = True
                    if not hit and r["cue_text"] and ctx:
                        toks = _cue_tokens(r["cue_text"])
                        if toks:
                            matched = sum(1 for t in toks if t in ctx)
                            hit = matched >= max(1, (len(toks) + 1) // 2)
                    if hit:
                        await conn.execute(
                            "UPDATE prospective_memory SET fired_at = $2 WHERE id = $1;",
                            r["id"], ts,
                        )
                        fired.append(dict(r))
                        log_atomic(f"E27 intention FIRED: '{r['intent']}'")
        except Exception as e:
            log_error(f"check_cues failed: {e}")
    return fired


async def complete_intention(intention_id: UUID) -> None:
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                "UPDATE prospective_memory SET is_done = TRUE WHERE id = $1;", intention_id)
    except Exception as e:
        log_error(f"complete_intention failed: {e}")


async def list_intentions(user_id: str, include_done: bool = False) -> List[Dict[str, Any]]:
    try:
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, intent, cue_text, cue_kind, trigger_at, fired_at, is_done, created_at
                FROM prospective_memory
                WHERE user_id = $1 AND ($2::bool OR is_done = FALSE)
                ORDER BY created_at DESC;
                """,
                user_id, include_done,
            )
            return [dict(r) for r in rows]
    except Exception as e:
        log_error(f"list_intentions failed: {e}")
        return []
