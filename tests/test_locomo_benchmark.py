"""LOCOMO-style benchmark: seed the multi-session fixture, run 10 eval cases
against the deterministic retrieval layer (search_facts). Zero-LLM, on medhas_test.

Run: POSTGRES_DB=medhas_test python -m pytest tests/test_locomo_benchmark.py -q
"""

import uuid as _uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio

from tests.locomo_fixture import FACTS, CASES, SESSIONS  # noqa: E402


def _uid() -> str:
    return "bench_" + _uuid.uuid4().hex[:8]


async def _seed_all(conn, user_id):
    from infrastructure.llm.embedding_provider import FastEmbeddingProvider
    import hashlib

    provider = FastEmbeddingProvider()
    for label, text, belief in FACTS:
        vf = SESSIONS.get(label, "2024-01-01")
        dt = datetime.fromisoformat(vf).replace(tzinfo=timezone.utc)
        vec = await provider.embed_text(text)
        vec_str = "[" + ",".join(str(x) for x in vec) + "]"
        h = hashlib.md5(text.strip().encode()).hexdigest()
        await conn.execute(
            """INSERT INTO atomic_facts
                  (user_id, fact_text, embedding, is_active, content_hash,
                   memory_type, metadata, valid_from, belief_confidence)
                VALUES ($1,$2,$3::vector,TRUE,$4,'semantic','{}',$5,$6)""",
            user_id, text, vec_str, h, dt, belief,
        )


async def test_locomo_benchmark_suite():
    from infrastructure.db import DatabasePool
    from memory.atomic.search_facts import search_facts
    from agi.eval import EvalCase, run_eval_suite

    uid = _uid()
    async with DatabasePool.acquire() as c:
        await _seed_all(c, uid)
        cases = [
            EvalCase(query=q, expects_fact=exp, should_abstain=False)
            for (q, exp, _kind) in CASES
        ]
        metrics = await run_eval_suite(
            uid, cases, recall_fn=lambda u, q, **kw: search_facts(u, q, limit=5)
        )
        await c.execute("DELETE FROM atomic_facts WHERE user_id=$1", uid)

    print("\nLOCOMO BENCHMARK:", metrics)
    assert metrics["total"] == len(CASES), "benchmark did not run all cases"
    # At least 8/10 (80%) retrieval recall on this deterministic fixture.
    assert metrics["passed"] >= 8, f"benchmark below threshold {metrics}"
    assert 0.0 <= metrics["accuracy"] <= 1.0
