"""Open relation vocabulary (schema evolution).

The memory system does NOT use a frozen, hard-coded list of relation types as its
primary extractor. Relations are discovered at runtime by the LLM (open extraction) and
persisted here, so the vocabulary *grows* instead of being capped at a fixed enum.

`agi.entities.RELATION_VERBS` is only the OFFLINE fallback seed — it bootstraps this
table (`source='seed'`) so the deterministic heuristic has a starting vocabulary, but any
relation the LLM emits (online) or the heuristic matches (offline) is recorded with
`source='extracted'/'inferred'` and counted. This matches the field consensus
(Graphiti/Zep, Mem0, Letta) where relation names are free-form and the edge-type set
evolves with the data.
"""

from typing import List, Optional, Set

from infrastructure.db import DatabasePool
from utils import log_error


async def record_relation(user_id: str, relation: str, source: str = "extracted") -> None:
    """Persist (or bump the count of) a discovered relation type for this user.

    Open/vocabulary: any non-empty relation string is accepted. This is what makes the
    schema evolve instead of being hard-coded.
    """
    rel = (relation or "").strip().upper()
    if not rel or len(rel) > 255:
        return
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO relation_types (user_id, relation, usage_count, source)
                VALUES ($1, $2, 1, $3)
                ON CONFLICT (user_id, relation) DO UPDATE
                    SET usage_count = relation_types.usage_count + 1
                """,
                user_id, rel, source,
            )
    except Exception as e:
        # Never let vocabulary bookkeeping break ingestion.
        log_error(f"record_relation skipped: {e}")


async def known_relations(user_id: str) -> Set[str]:
    """Return the currently-known relation vocabulary for this user (seeds + discovered)."""
    try:
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT relation FROM relation_types WHERE user_id=$1 ORDER BY usage_count DESC",
                user_id,
            )
            return {r["relation"] for r in rows}
    except Exception as e:
        log_error(f"known_relations skipped: {e}")
        return set()


async def seed_default_relations(user_id: str, verbs: dict, rule_relations: Optional[List[str]] = None) -> None:
    """Bootstrap the vocabulary table from the offline fallback seed (RELATION_VERBS)
    and the reasoning-rule relations. Idempotent (ON CONFLICT DO NOTHING semantics via
    the PK + usage_count bump is harmless)."""
    from agi.entities import RELATION_VERBS
    seed_rels = set(RELATION_VERBS.values())
    if rule_relations:
        seed_rels.update(rule_relations)
    for rel in sorted(seed_rels):
        try:
            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO relation_types (user_id, relation, usage_count, source)
                    VALUES ($1, $2, 1, 'seed')
                    ON CONFLICT (user_id, relation) DO NOTHING
                    """,
                    user_id, rel,
                )
        except Exception:
            pass


__all__ = ["record_relation", "known_relations", "seed_default_relations"]
