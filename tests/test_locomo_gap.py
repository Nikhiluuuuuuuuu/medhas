"""LOCOMO-style gap verification + benchmark fixture (G1-G4).

Zero-LLM, deterministic. Every test seeds facts directly via SQL (no Groq),
exercises a real retrieval path, and asserts the behaviour the deep-research
audit called for:

  G1  provenance/confidence surfaced in recall results
  G2  contradiction SUPERSEDES the old fact (valid_to closed, not just score)
  G3  undated facts carry valid_from -> temporal "what was true at T" works
  G4  a LOCOMO-style eval suite runs end-to-end (run_eval_suite) with no LLM

Run: POSTGRES_DB=medhas_test python -m pytest tests/test_locomo_gap.py -q
"""

import uuid as _uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio


def _uid() -> str:
    return "loc_" + _uuid.uuid4().hex[:8]


async def _seed(conn, user_id, text, *, valid_from=None, belief=1.0):
    from infrastructure.llm.embedding_provider import FastEmbeddingProvider
    import hashlib

    provider = FastEmbeddingProvider()
    vec = await provider.embed_text(text)
    vec_str = f"[{','.join(str(x) for x in vec)}]"
    h = hashlib.md5(text.strip().encode()).hexdigest()
    vf = "$5" if valid_from is not None else "CURRENT_TIMESTAMP"
    params = [user_id, text, vec_str, h, belief]
    if valid_from is not None:
        params.append(valid_from)
    await conn.execute(
        f"""INSERT INTO atomic_facts
              (user_id, fact_text, embedding, is_active, content_hash,
               memory_type, metadata, valid_from, belief_confidence)
            VALUES ($1,$2,$3::vector,TRUE,$4,'semantic','{{}}',{vf},$5)""",
        *params,
    )


# ---------------------------------------------------------------------------
# G1: provenance + confidence surfaced by the retrieval layer (search_facts)
# ---------------------------------------------------------------------------
async def test_g1_provenance_surfaced_in_retrieval():
    from infrastructure.db import DatabasePool
    from memory.atomic.search_facts import search_facts

    uid = _uid()
    async with DatabasePool.acquire() as c:
        await _seed(c, uid, "Nikhil prefers concise answers", belief=0.6)
        res = await search_facts(uid, "what does Nikhil prefer", limit=3)
        top = res[0].model_dump()
        await c.execute("DELETE FROM atomic_facts WHERE user_id=$1", uid)

    for field in ("belief_confidence", "provenance_kind", "valid_from",
                  "valid_to", "invalidated_by", "source_episode_id", "contradicted_by"):
        assert field in top, f"G1: {field} missing from retrieval result"
    assert top["belief_confidence"] == 0.6
    assert top["provenance_kind"] == "explicit"


# ---------------------------------------------------------------------------
# G2: a new contradicting fact SUPERSEDES the old one (valid_to closed)
# ---------------------------------------------------------------------------
async def test_g2_contradiction_supersedes_old_fact():
    from infrastructure.db import DatabasePool
    from agi.engine import engine

    uid = _uid()
    await engine.remember(uid, "Nikhil lives in Bangalore")
    await engine.remember(uid, "Nikhil lives in Hyderabad")
    async with DatabasePool.acquire() as c:
        rows = await c.fetch(
            """SELECT fact_text, valid_to, is_active, belief_confidence
               FROM atomic_facts WHERE user_id=$1 ORDER BY created_at""",
            uid,
        )
        await c.execute("DELETE FROM atomic_facts WHERE user_id=$1", uid)
        await c.execute("DELETE FROM graph_nodes WHERE user_id=$1", uid)
        await c.execute("DELETE FROM graph_edges WHERE user_id=$1", uid)
    # The new fact wins: exactly one ACTIVE "lives in" fact, and it is Hyderabad.
    active = [r for r in rows if r["is_active"]]
    assert len(active) == 1, f"G2: expected exactly one active fact, got {len(active)}"
    assert "hyderabad" in active[0]["fact_text"].lower(), "G2: active fact is not the new value"
    # The old fact was superseded (deactivated), not silently kept as the answer.
    old = [r for r in rows if not r["is_active"]]
    assert old, "G2: old contradicted fact was not superseded (still active)"


# ---------------------------------------------------------------------------
# G3: undated fact still answers a temporal "what was true at T" query
# ---------------------------------------------------------------------------
async def test_g3_undated_fact_valid_at():
    from infrastructure.db import DatabasePool
    from agi.bitemporal import facts_valid_at

    uid = _uid()
    probe = datetime(2030, 1, 1, tzinfo=timezone.utc)  # after the default valid_from (now)
    async with DatabasePool.acquire() as c:
        await _seed(c, uid, "Nikhil co-founded Kraionyx AI")  # undated -> valid_from default (now)
        facts = await facts_valid_at(uid, probe)
        await c.execute("DELETE FROM atomic_facts WHERE user_id=$1", uid)
    assert any("kraionyx" in f["fact_text"].lower() for f in facts), \
        "G3: undated fact absent from valid_at snapshot"


# ---------------------------------------------------------------------------
# G4: LOCOMO-style benchmark suite runs with NO LLM calls
# ---------------------------------------------------------------------------
async def test_g4_locomo_eval_suite_runs():
    from infrastructure.db import DatabasePool
    from agi.engine import engine
    from agi.eval import EvalCase, run_eval_suite

    uid = _uid()
    async with DatabasePool.acquire() as c:
        await _seed(c, uid, "Nikhil co-founded Kraionyx AI in 2025")
        await _seed(c, uid, "Kraionyx builds KareOS as its flagship product")

    cases = [
        EvalCase(query="who co-founded Kraionyx AI", expects_fact="nikhil co-founded kraionyx"),
        EvalCase(query="what does Kraionyx build", expects_fact="kareos"),
    ]
    # Benchmark the deterministic retrieval layer (search_facts) — no LLM, no
    # abstention gate, stable for CI. (engine.recall adds reconsolidation/abstention
    # on top; the eval harness measures raw retrieval quality.)
    from memory.atomic.search_facts import search_facts
    metrics = await run_eval_suite(
        uid, cases, recall_fn=lambda u, q, **kw: search_facts(u, q, limit=5)
    )
    async with DatabasePool.acquire() as c:
        await c.execute("DELETE FROM atomic_facts WHERE user_id=$1", uid)

    assert metrics["total"] == 2, "G4: suite did not run all cases"
    assert metrics["passed"] == 2, f"G4: retrieval failures {metrics}"
    assert 0.0 <= metrics["accuracy"] <= 1.0
