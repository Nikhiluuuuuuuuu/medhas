"""Open relation vocabulary (schema evolution).

The memory system does NOT use a frozen, hard-coded list of relation types as its
primary extractor. Relations are discovered at runtime by the LLM (open extraction) and
persisted here, so the vocabulary *grows* instead of being capped at a fixed enum.

`agi.entities.RELATION_VERBS` (a closed verb dictionary) was removed. The vocabulary is
now seeded only from caller-provided relations (e.g. the reasoning-rule axioms) and
otherwise grows purely from relations the LLM discovers at runtime, recorded here with
`source='extracted'/'inferred'` and counted. This matches the field consensus
(Graphiti/Zep, Mem0, Letta) where relation names are free-form and the edge-type set
evolves with the data.
"""

from typing import List, Optional, Set

from medhas.storage import DatabasePool
from medhas.utils import log_error


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


async def seed_default_relations(user_id: str, rule_relations: Optional[List[str]] = None) -> None:
    """Optionally bootstrap the vocabulary from a caller-provided set of relations
    (e.g. the reasoning-rule relations). There is NO hard-coded verb list: the
    vocabulary is meant to grow from relations the LLM actually discovers. If no
    relations are supplied, this is a no-op and the set starts empty.
    """
    seed_rels = set(rule_relations or [])
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
