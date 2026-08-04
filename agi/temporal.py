"""G3 — Genuine temporal recall.

Fixes the silent temporal gap: facts now carry valid_from (set in insert_fact), so
facts_valid_at(user_id, at) returns a real "what was true at T" snapshot. This module
adds intent detection for before/after questions and a resolver that answers them by
anchoring to the referenced entity's current fact time and querying the valid-time lattice.

Example: "where did Nikhil live before Hyderabad"
  1. detect "before" + anchor entity "Hyderabad"
  2. find the fact about living in Hyderabad -> its valid_from = t_hyd
  3. facts_valid_at(user_id, t_hyd - epsilon) -> the prior residence (Bangalore)
"""

from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timezone, timedelta

from infrastructure.db import DatabasePool
from utils import measure_latency, log_atomic, log_error

from agi.bitemporal import facts_valid_at
from agi.entities import query_entities


def detect_temporal_intent(query: str) -> Optional[Dict[str, str]]:
    """Return {'mode': 'before'|'after', 'anchor': <entity token>} or None."""
    q = query.lower()
    for mode, marker in (("before", "before"), ("after", "after"),
                         ("before", "prior to"), ("after", "since"),
                         ("before", "previously"), ("before", "used to")):
        if marker in q:
            # anchor = first capitalized entity after the marker
            tail = query.split(marker, 1)[1]
            ents = query_entities(tail)
            if ents:
                return {"mode": mode, "anchor": ents[0]}
            # fallback: first non-stopword token after marker
            toks = [t for t in tail.replace(",", " ").split() if t.lower() not in
                    {"the", "a", "an", "in", "to", "at", "of", "on", "for"}]
            if toks:
                return {"mode": mode, "anchor": toks[0].strip(".,!?:;\"'()[]")}
    return None


async def recall_before_after(
    user_id: str,
    query: str,
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Answer a before/after temporal query using the valid-time lattice."""
    async with measure_latency("agi.temporal.recall_before_after"):
        try:
            intent = detect_temporal_intent(query)
            if intent is None:
                return []
            anchor = intent["anchor"]
            mode = intent["mode"]

            # find the anchor fact's valid_from (latest mention of the anchor entity,
            # active OR inactive — it is only a temporal reference point; the answer
            # facts are still filtered by is_active in facts_valid_at below)
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, valid_from, created_at FROM atomic_facts
                    WHERE user_id = $1
                      AND LOWER(fact_text) LIKE '%' || LOWER($2) || '%'
                    ORDER BY valid_from DESC NULLS LAST, created_at DESC
                    LIMIT 1;
                    """,
                    user_id, anchor,
                )
            if not row:
                return []
            anchor_id = row["id"]
            anchor_time = row["valid_from"] or row["created_at"] or datetime.now(timezone.utc)
            if mode == "before":
                probe = anchor_time - timedelta(seconds=1)
            else:
                probe = anchor_time + timedelta(seconds=1)

            facts = await facts_valid_at(user_id, probe, limit=limit)
            # exclude the anchor fact itself (matched by id, not by substring — the
            # answer fact legitimately mentions the anchor entity)
            return [f for f in facts if f.get("id") != anchor_id][:limit]
        except Exception as e:
            log_error(f"recall_before_after skipped: {e}")
            return []


__all__ = ["detect_temporal_intent", "recall_before_after"]
