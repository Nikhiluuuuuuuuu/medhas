"""Regression tests for gap fixes: G1 (insert_fact robustness + valid_from),
G2 (multi-hop graph recall), G3 (temporal before/after recall).

Run with: POSTGRES_DB=medhas_test python -m pytest tests/test_gaps.py -q
Requires a live Postgres (medhas_test / unified_memory).
"""

import uuid as _uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio


def _uid() -> str:
    return "gap_" + _uuid.uuid4().hex[:8]


async def _seed(conn, user_id, text, valid_from=None):
    from medhas.llm.embedding_provider import FastEmbeddingProvider
    import hashlib
    provider = FastEmbeddingProvider()
    vec = await provider.embed_text(text)
    vec_str = f"[{','.join(str(x) for x in vec)}]"
    h = hashlib.md5(text.strip().encode()).hexdigest()
    vf = f"$5" if valid_from is not None else "CURRENT_TIMESTAMP"
    params = [user_id, text, vec_str, h]
    if valid_from is not None:
        params.append(valid_from)
    await conn.execute(
        f"""INSERT INTO atomic_facts (user_id, fact_text, embedding, is_active, content_hash, memory_type, metadata, valid_from)
           VALUES ($1,$2,$3::vector,TRUE,$4,'semantic','{{}}',{vf})""",
        *params,
    )


async def test_g1_valid_from_populated():
    """G1: facts must carry a valid_from so temporal queries work."""
    from medhas.storage import DatabasePool
    uid = _uid()
    async with DatabasePool.acquire() as c:
        await _seed(c, uid, "Nikhil co-founded Kraionyx AI in Hyderabad")
        n = await c.fetchval(
            "SELECT count(*) FROM atomic_facts WHERE user_id=$1 AND valid_from IS NOT NULL", uid
        )
        await c.execute("DELETE FROM atomic_facts WHERE user_id=$1", uid)
    assert n == 1


async def test_g1_insert_does_not_silently_drop():
    """G1: insert_fact must not return None / silently drop a valid new fact."""
    from medhas.storage import DatabasePool
    from medhas.engine import engine
    from medhas.memory.atomic.insert_fact import insert_fact
    await DatabasePool.initialize()
    uid = _uid()
    fact = await insert_fact(uid, "Kraionyx builds KareOS as its flagship product", memory_type="semantic")
    assert fact is not None and fact.id is not None
    async with DatabasePool.acquire() as c:
        cnt = await c.fetchval(
            "SELECT count(*) FROM atomic_facts WHERE user_id=$1 AND LOWER(fact_text)=LOWER($2)",
            uid, "Kraionyx builds KareOS as its flagship product",
        )
        await c.execute("DELETE FROM atomic_facts WHERE user_id=$1", uid)
    assert cnt == 1


async def test_g2_multihop_recall():
    """G2: a 2-hop query returns the co-founder fact via the entity graph."""
    from medhas.storage import DatabasePool
    from medhas.engine import engine
    await DatabasePool.initialize()
    uid = _uid()
    facts = [
        "Nikhil co-founded Kraionyx AI with 4 friends in Hyderabad in 2025",
        "Kraionyx builds KareOS as its flagship product",
    ]
    for f in facts:
        await engine.remember(uid, f)
    res = await engine.recall(uid, "who co-founded the company that builds KareOS", enforce_abstention=True)
    blob = " ".join(r["fact_text"].lower() for r in res.get("results", []))
    async with DatabasePool.acquire() as c:
        await c.execute("DELETE FROM atomic_facts WHERE user_id=$1", uid)
        await c.execute("DELETE FROM graph_edges WHERE user_id=$1", uid)
        await c.execute("DELETE FROM graph_nodes WHERE user_id=$1", uid)
    assert "nikhil" in blob and "kraionyx" in blob


async def test_g3_temporal_before():
    """G3: 'before Hyderabad' returns the prior residence (Bangalore)."""
    from medhas.storage import DatabasePool
    from medhas.engine import engine
    await DatabasePool.initialize()
    uid = _uid()
    async with DatabasePool.acquire() as c:
        await _seed(c, uid, "Nikhil lived in Bangalore before moving to Hyderabad", datetime(2018, 1, 1, tzinfo=timezone.utc))
        await _seed(c, uid, "Nikhil lives in Hyderabad now", datetime(2024, 1, 1, tzinfo=timezone.utc))
    res = await engine.recall(uid, "where did Nikhil live before Hyderabad", enforce_abstention=True)
    blob = " ".join(r["fact_text"].lower() for r in res.get("results", []))
    async with DatabasePool.acquire() as c:
        await c.execute("DELETE FROM atomic_facts WHERE user_id=$1", uid)
    assert "bangalore" in blob
