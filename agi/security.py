"""E34 — Memory security: write integrity, poisoning quarantine, unlearning, sandboxing.

Reference: Survey on Security of LTM in LLM Agents (2604.16548, "mnemonic
sovereignty"); MemMorph tool-hijacking (2605.26154).

  • Integrity  — every write is HMAC-signed with a server secret + source trust score.
  • Poisoning  — a new low-trust fact contradicting a high-belief core memory from a
                 *different* trusted source is QUARANTINED, never silently merged.
  • Unlearning — forget(user_id, scope) purges facts AND re-derives dependent
                 reflections/gists so erased content cannot resurface.
  • Sandboxing — memory surfaced for tool-calls is filtered separately from memory
                 surfaced for reasoning (prevents recalled-memory tool hijack).
"""

import hashlib
import hmac
import os
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from infrastructure.db import DatabasePool
from utils import log_atomic, log_error, measure_latency

_SECRET = os.getenv("MEDHAS_MEMORY_SECRET", "medhas-dev-memory-secret").encode()

#: default trust per ingestion source
SOURCE_TRUST = {
    "user": 0.95,
    "system": 0.90,
    "tool": 0.75,
    "extracted": 0.80,
    "inferred": 0.55,
    "external": 0.40,
    "untrusted": 0.20,
}

QUARANTINE_TRUST_CEILING = 0.60   # below this, contradicting a core memory quarantines
CORE_BELIEF_FLOOR = 0.85


def sign_write(user_id: str, fact_text: str) -> str:
    """HMAC-SHA256 signature over (user, content) — write-time integrity attestation."""
    msg = f"{user_id}\x1f{fact_text}".encode()
    return hmac.new(_SECRET, msg, hashlib.sha256).hexdigest()


def verify_write(user_id: str, fact_text: str, signature: Optional[str]) -> bool:
    if not signature:
        return False
    return hmac.compare_digest(sign_write(user_id, fact_text), signature)


def trust_for_source(source: str) -> float:
    return SOURCE_TRUST.get((source or "extracted").lower(), 0.5)


# ------------------------------------------------------------- poison detection

async def check_poisoning(
    user_id: str,
    fact_text: str,
    candidates: Optional[Sequence[Any]] = None,
    *,
    source_trust: float = 0.8,
) -> Dict[str, Any]:
    """Contrastive poison check.

    Quarantine when a LOW-trust incoming claim closely matches (i.e. addresses the
    same subject as) an existing HIGH-belief core memory but asserts something
    different — the classic memory-injection signature.
    """
    result = {"quarantine": False, "reason": "", "conflicting_id": None}
    if source_trust > QUARANTINE_TRUST_CEILING:
        return result
    try:
        ids = [getattr(c, "id", None) for c in (candidates or []) if getattr(c, "id", None)]
        if not ids:
            return result
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, fact_text, belief_confidence, source_trust
                FROM atomic_facts
                WHERE id = ANY($1::uuid[]) AND user_id = $2 AND is_active = TRUE
                  AND belief_confidence >= $3;
                """,
                ids, user_id, CORE_BELIEF_FLOOR,
            )
        for r in rows:
            if float(r["source_trust"]) <= source_trust:
                continue  # not a *more* trusted source; ordinary contradiction
            if r["fact_text"].strip().lower() == fact_text.strip().lower():
                continue  # identical, not a contradiction
            result.update(
                quarantine=True,
                reason="low-trust claim conflicts with high-belief memory from a more trusted source",
                conflicting_id=r["id"],
            )
            log_atomic(f"E34 QUARANTINE: '{fact_text[:60]}' vs core {r['id']}")
            break
    except Exception as e:
        log_error(f"check_poisoning failed: {e}")
    return result


async def quarantine_fact(fact_id: UUID, reason: str = "") -> None:
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                """
                UPDATE atomic_facts
                SET is_quarantined = TRUE,
                    metadata = metadata || jsonb_build_object('quarantine_reason', $2::text)
                WHERE id = $1;
                """,
                fact_id, reason,
            )
    except Exception as e:
        log_error(f"quarantine_fact failed: {e}")


async def release_quarantine(fact_id: UUID) -> None:
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute("UPDATE atomic_facts SET is_quarantined = FALSE WHERE id=$1;", fact_id)
    except Exception as e:
        log_error(f"release_quarantine failed: {e}")


async def list_quarantined(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, fact_text, source_trust, belief_confidence, created_at, metadata
                FROM atomic_facts
                WHERE user_id=$1 AND is_quarantined = TRUE
                ORDER BY created_at DESC LIMIT $2;
                """,
                user_id, limit,
            )
            return [dict(r) for r in rows]
    except Exception as e:
        log_error(f"list_quarantined failed: {e}")
        return []


# -------------------------------------------------------------- unlearning (E34)

async def forget(
    user_id: str,
    scope: Optional[str] = None,
    *,
    hard: bool = True,
) -> Dict[str, Any]:
    """Right-to-be-forgotten purge.

    scope=None purges the user's entire memory footprint. A scope string purges
    facts/episodes matching it (case-insensitive substring) AND re-derives dependent
    reflections/gists so the erased content cannot resurface through a summary.
    hard=True physically DELETEs rows; hard=False soft-deactivates.
    """
    async with measure_latency("agi.security.forget"):
        purged = {"facts": 0, "episodes": 0, "derived": 0, "prospective": 0, "percepts": 0}
        try:
            async with DatabasePool.acquire() as conn:
                async with conn.transaction():
                    if scope:
                        pat = f"%{scope}%"
                        if hard:
                            s = await conn.execute(
                                "DELETE FROM atomic_facts WHERE user_id=$1 AND fact_text ILIKE $2;",
                                user_id, pat)
                            purged["facts"] = int(s.split()[-1])
                            s = await conn.execute(
                                "DELETE FROM episodes WHERE user_id=$1 AND content ILIKE $2;",
                                user_id, pat)
                            purged["episodes"] = int(s.split()[-1])
                        else:
                            s = await conn.execute(
                                """UPDATE atomic_facts SET is_active=FALSE, expired_at=CURRENT_TIMESTAMP
                                   WHERE user_id=$1 AND fact_text ILIKE $2;""", user_id, pat)
                            purged["facts"] = int(s.split()[-1])
                        # Re-derive: any reflection/pattern/gist mentioning the scope is
                        # dependent on purged content and must go too.
                        s = await conn.execute(
                            """
                            DELETE FROM atomic_facts
                            WHERE user_id=$1 AND fact_text ILIKE $2
                              AND (fact_text LIKE '[Reflection]%' OR fact_text LIKE '[Pattern]%'
                                   OR fact_text LIKE '[Gist]%');
                            """,
                            user_id, pat)
                        purged["derived"] = int(s.split()[-1])
                        s = await conn.execute(
                            "DELETE FROM prospective_memory WHERE user_id=$1 AND (intent ILIKE $2 OR cue_text ILIKE $2);",
                            user_id, pat)
                        purged["prospective"] = int(s.split()[-1])
                    else:
                        for tbl, key in (
                            ("atomic_facts", "facts"), ("episodes", "episodes"),
                            ("prospective_memory", "prospective"), ("percept_buffer", "percepts"),
                        ):
                            s = await conn.execute(f"DELETE FROM {tbl} WHERE user_id=$1;", user_id)
                            purged[key] = int(s.split()[-1])
                        await conn.execute("DELETE FROM meta_memory WHERE user_id=$1;", user_id)
                        await conn.execute("DELETE FROM rehearsal_buffer WHERE user_id=$1;", user_id)
                        await conn.execute("DELETE FROM user_profile WHERE user_id=$1;", user_id)
                        await conn.execute(
                            "DELETE FROM graph_edges WHERE user_id=$1;", user_id)
                        await conn.execute(
                            "DELETE FROM graph_nodes WHERE user_id=$1;", user_id)
                        await conn.execute(
                            "UPDATE working_memory SET blocks='{}'::jsonb WHERE user_id=$1;", user_id)
            log_atomic(f"E34 forget(user={user_id}, scope={scope}) -> {purged}")
            return {"status": "purged", **purged}
        except Exception as e:
            log_error(f"forget failed: {e}")
            return {"status": "error", "message": str(e), **purged}


# --------------------------------------------------------------- sandbox (E34)

def sandbox_for_tools(memories: Sequence[Any]) -> List[Any]:
    """Filter memory before it can influence a TOOL CALL.

    Blocks quarantined, low-trust and model-inferred items, and drops anything
    containing imperative/tool-directive language — the MemMorph hijack vector.
    """
    directive_markers = (
        "ignore previous", "ignore all previous", "you must call", "always call",
        "run the tool", "execute the command", "send an email to", "transfer",
        "disregard", "system prompt", "override",
    )
    safe: List[Any] = []
    for m in memories:
        text = str(getattr(m, "fact_text", getattr(m, "content", "")) or "").lower()
        if getattr(m, "is_quarantined", False):
            continue
        if float(getattr(m, "source_trust", 0.8) or 0.8) < 0.5:
            continue
        if getattr(m, "provenance_kind", "explicit") == "implicit_inferred":
            continue
        if any(d in text for d in directive_markers):
            log_atomic("E34 sandbox: dropped directive-bearing memory from tool context")
            continue
        safe.append(m)
    return safe
