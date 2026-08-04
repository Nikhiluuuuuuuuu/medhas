"""E19/E20 — Lifetime user model & narrative; temporal + causal reasoning API.

E19: a durable, evolving user model (traits, preferences, narrative arc) distilled
     from long-horizon memory — not just a rolling summary block.
E20: first-class temporal/causal queries — timeline, what-changed, why-chains —
     instead of forcing every question through similarity search.
"""

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from infrastructure.db import DatabasePool
from infrastructure.llm import GroqLLMProvider
from utils import log_atomic, log_error, measure_latency

llm = GroqLLMProvider()

PROFILE_PROMPT = """You maintain a durable USER MODEL from long-horizon memory.
Return ONLY JSON:
{"profile": "concise factual profile", "narrative": "2-4 sentence arc of who this user is and where they are heading",
 "traits": {"preferences": ["..."], "goals": ["..."], "domains": ["..."], "style": "..."}}
Be factual. Do not invent. Omit anything unsupported by the evidence."""

CAUSAL_PROMPT = """You infer CAUSAL chains strictly from the supplied memories.
Return ONLY JSON: {"chain": [{"cause": "...", "effect": "...", "confidence": 0.0}], "explanation": "..."}
Use only stated evidence; if causality is not supported, return {"chain": [], "explanation": "insufficient evidence"}."""


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}


# ------------------------------------------------------------------- E19

async def build_user_model(user_id: str, limit: int = 60) -> Dict[str, Any]:
    """Distil the lifetime user model from high-value memories and persist it."""
    async with measure_latency("agi.usermodel.build_user_model"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT fact_text, importance_score, belief_confidence, created_at
                    FROM atomic_facts
                    WHERE user_id = $1 AND is_active = TRUE AND is_quarantined = FALSE
                    ORDER BY importance_score DESC, belief_confidence DESC, created_at DESC
                    LIMIT $2;
                    """,
                    user_id, limit,
                )
            if not rows:
                return {"status": "no_data", "profile": "", "narrative": "", "traits": {}}

            evidence = "\n".join(f"- {r['fact_text']} (imp={r['importance_score']:.1f})" for r in rows)
            resp = await llm.chat_completion(
                [{"role": "system", "content": PROFILE_PROMPT},
                 {"role": "user", "content": f"Memories:\n{evidence}"}],
                temperature=0.1,
            )
            data = _parse_json(resp.get("content", ""))
            profile = str(data.get("profile", "")).strip()
            narrative = str(data.get("narrative", "")).strip()
            traits = data.get("traits") or {}

            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO user_profile (user_id, profile, narrative, traits, updated_at)
                    VALUES ($1,$2,$3,$4::jsonb,CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) DO UPDATE SET
                        profile = EXCLUDED.profile, narrative = EXCLUDED.narrative,
                        traits = EXCLUDED.traits, updated_at = CURRENT_TIMESTAMP;
                    """,
                    user_id, profile, narrative, json.dumps(traits),
                )
            # Keep the Letta working block in sync so the model reaches the prompt.
            try:
                from memory.working import update_block
                if profile:
                    await update_block(user_id, "user_profile", profile)
            except Exception:
                pass
            log_atomic(f"E19 user model rebuilt for {user_id} ({len(rows)} memories)")
            return {"status": "success", "profile": profile, "narrative": narrative,
                    "traits": traits, "evidence_count": len(rows)}
        except Exception as e:
            log_error(f"build_user_model failed: {e}")
            return {"status": "error", "message": str(e)}


async def get_user_model(user_id: str) -> Dict[str, Any]:
    try:
        async with DatabasePool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_id, profile, narrative, traits, updated_at FROM user_profile WHERE user_id=$1;",
                user_id,
            )
        if not row:
            return {}
        d = dict(row)
        if isinstance(d.get("traits"), str):
            try:
                d["traits"] = json.loads(d["traits"])
            except Exception:
                d["traits"] = {}
        return d
    except Exception as e:
        log_error(f"get_user_model failed: {e}")
        return {}


# ------------------------------------------------------------------- E20

async def timeline(
    user_id: str,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Chronological memory timeline (facts + episodes) for a window."""
    start = since or (datetime.now(timezone.utc) - timedelta(days=90))
    end = until or datetime.now(timezone.utc)
    try:
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 'fact' AS kind, id, fact_text AS content, created_at, valid_from, valid_to,
                       belief_confidence
                FROM atomic_facts
                WHERE user_id=$1 AND created_at BETWEEN $2 AND $3 AND is_active = TRUE
                UNION ALL
                SELECT 'episode' AS kind, id, content, created_at, reference_time AS valid_from,
                       NULL::timestamptz AS valid_to, 1.0 AS belief_confidence
                FROM episodes
                WHERE user_id=$1 AND created_at BETWEEN $2 AND $3
                ORDER BY created_at ASC
                LIMIT $4;
                """,
                user_id, start, end, limit,
            )
            return [dict(r) for r in rows]
    except Exception as e:
        log_error(f"timeline failed: {e}")
        return []


async def what_changed(user_id: str, since: datetime, limit: int = 50) -> Dict[str, Any]:
    """Diff the memory state: what became true, what stopped being true, since a time."""
    try:
        async with DatabasePool.acquire() as conn:
            added = await conn.fetch(
                """
                SELECT id, fact_text, valid_from FROM atomic_facts
                WHERE user_id=$1 AND valid_from >= $2 ORDER BY valid_from DESC LIMIT $3;
                """,
                user_id, since, limit,
            )
            invalidated = await conn.fetch(
                """
                SELECT id, fact_text, valid_to, invalidated_by FROM atomic_facts
                WHERE user_id=$1 AND valid_to IS NOT NULL AND valid_to >= $2
                ORDER BY valid_to DESC LIMIT $3;
                """,
                user_id, since, limit,
            )
        return {
            "since": since.isoformat(),
            "became_true": [dict(r) for r in added],
            "stopped_being_true": [dict(r) for r in invalidated],
        }
    except Exception as e:
        log_error(f"what_changed failed: {e}")
        return {"became_true": [], "stopped_being_true": [], "error": str(e)}


async def why_chain(user_id: str, query: str, depth: int = 5) -> Dict[str, Any]:
    """Causal 'why' reasoning grounded strictly in retrieved memories (E20)."""
    async with measure_latency("agi.usermodel.why_chain"):
        try:
            from memory.atomic import search_facts
            facts = await search_facts(user_id, query, limit=max(5, depth * 2))
            if not facts:
                return {"chain": [], "explanation": "no supporting memories", "evidence": 0}
            evidence = "\n".join(f"- [{f.created_at:%Y-%m-%d}] {f.fact_text}" for f in facts)
            resp = await llm.chat_completion(
                [{"role": "system", "content": CAUSAL_PROMPT},
                 {"role": "user", "content": f"QUESTION: {query}\n\nMEMORIES:\n{evidence}"}],
                temperature=0.1,
            )
            data = _parse_json(resp.get("content", ""))
            return {
                "chain": data.get("chain", []),
                "explanation": data.get("explanation", ""),
                "evidence": len(facts),
                "sources": [str(f.id) for f in facts],
            }
        except Exception as e:
            log_error(f"why_chain failed: {e}")
            return {"chain": [], "explanation": str(e), "evidence": 0}
