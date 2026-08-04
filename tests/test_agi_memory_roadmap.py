"""End-to-end + unit tests for the AGI memory roadmap (E1–E37).

These exercises run against the real Postgres test database (medhas_test) wired via
POSTGRES_DB. They verify the additive modules work, not just import.
"""

import os
import pytest
import pytest_asyncio
import asyncio

from medhas.storage import DatabasePool, initialize_schema
from medhas.engine import engine
from medhas.memory.control.memory_types import MemoryType, route
from medhas.memory.control.admission import evaluate_admission
from medhas.platform.security import sign_write, verify_write
from medhas.memory.graph.bitemporal import bayesian_update
from medhas.memory.control.interference import interference_matrix, eviction_scores
from medhas.memory.operations.ingest import chunk_text
from medhas.metamemory.metacognitive import route_query
from medhas.platform.eval import temporal_consistency_check

UID = "agi_test_user"


@pytest_asyncio.fixture(autouse=True)
async def _db():
    # Pool + schema are session-scoped (conftest.py). This fixture only cleans the
    # shared user_id between tests so cases stay isolated without re-running DDL.
    yield
    async with DatabasePool.acquire() as conn:
        await conn.execute("DELETE FROM atomic_facts WHERE user_id = $1;", UID)
        await conn.execute("DELETE FROM episodes WHERE user_id = $1;", UID)
        await conn.execute("DELETE FROM prospective_memory WHERE user_id = $1;", UID)
        await conn.execute("DELETE FROM meta_memory WHERE user_id = $1;", UID)
        await conn.execute("DELETE FROM percept_buffer WHERE user_id = $1;", UID)
        await conn.execute("DELETE FROM eval_runs WHERE user_id = $1;", UID)
        await conn.execute("DELETE FROM memory_events WHERE user_id = $1;", UID)


def test_memory_type_routing():
    assert route("semantic") == "atomic_facts"
    assert route("prospective") == "prospective_memory"
    assert route("sensory") == "percept_buffer"
    assert MemoryType("affective") == MemoryType.AFFECTIVE


def test_admission_control():
    d_store = evaluate_admission("User's favorite language is Python", None, source_trust=0.95)
    assert d_store.action == "STORE"
    d_drop = evaluate_admission("ok", None, source_trust=0.2)
    assert d_drop.action == "DROP"


def test_bayesian_belief():
    p = bayesian_update(0.7, 0.75, supports=True)
    assert 0.7 < p <= 0.99
    p2 = bayesian_update(0.7, 0.75, supports=False)
    assert p2 < 0.7


def test_write_signature():
    s = sign_write(UID, "fact")
    assert verify_write(UID, "fact", s)
    assert not verify_write(UID, "fact", "tampered")


def test_chunking():
    text = ("Sentence one about the user. " * 400)
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1800 for c in chunks)


def test_eviction_scores():
    blocks = {"persona": "x", "user_profile": "y" * 300, "scratchpad": "z" * 900}
    scored = eviction_scores(blocks, query="user_profile")
    names = [n for n, _ in scored]
    assert names[0] == "user_profile" or "scratchpad" in names[:2]


def test_routing():
    r = route_query("who is connected to Alice and why did it change", recognized=False)
    assert r.use_graph is True
    r2 = route_query("what is my name", recognized=True)
    assert r2.strategy == "recognition"


async def test_engine_remember_recall():
    res = await engine.remember(UID, "Nikhil's favorite language is Python", source="user")
    assert res["status"] in ("stored", "quarantined")
    assert res.get("memory_type") == "semantic"
    rec = await engine.recall(UID, "what language does Nikhil like")
    assert rec["status"] in ("ok", "abstained")
    if rec["status"] == "ok":
        assert len(rec["results"]) >= 1


async def test_prospective_memory():
    from medhas.engine import add_intention, check_cues
    iid = await add_intention(UID, "Remind me to submit the report", cue_text="submit the report")
    assert iid is not None
    fired = await check_cues(UID, current_context="Remember to submit the report today")
    assert any(str(f["id"]) == str(iid) for f in fired)


async def test_bitemporal_contradiction():
    from medhas.engine import invalidate_fact
    from medhas.memory.atomic import insert_fact
    from datetime import datetime, timezone
    f1 = await insert_fact(UID, "The project deadline is March 1", memory_type="semantic")
    f2 = await insert_fact(UID, "The project deadline is April 15", memory_type="semantic")
    await invalidate_fact(f1.id, invalidated_by=f2.id,
                          valid_to=datetime.now(timezone.utc))
    async with DatabasePool.acquire() as conn:
        vt = await conn.fetchval("SELECT valid_to FROM atomic_facts WHERE id=$1;", f1.id)
    assert vt is not None


async def test_forgetting_and_protected():
    from medhas.engine import protect_core_memories, run_forgetting_sweep
    from medhas.memory.atomic import insert_fact
    # high-importance memory should become protected and survive forgetting
    f = await insert_fact(UID, "Core identity fact", memory_type="semantic")
    async with DatabasePool.acquire() as conn:
        await conn.execute(
            "UPDATE atomic_facts SET importance_score=9.5, belief_confidence=0.95 WHERE id=$1;",
            f.id)
    n = await protect_core_memories(UID)
    assert n >= 1
    sweep = await run_forgetting_sweep(UID)
    # core memory is protected -> not forgotten
    async with DatabasePool.acquire() as conn:
        active = await conn.fetchval(
            "SELECT is_active FROM atomic_facts WHERE id=$1;", f.id)
    assert active is True


async def test_sensory_buffer():
    from medhas.engine import buffer_percept, promote_percepts, list_buffer
    pid = await buffer_percept(UID, "Invoice total is 1042 USD from vendor Acme", modality="document")
    assert pid is not None
    buf = await list_buffer(UID)
    assert any(str(b["id"]) == str(pid) for b in buf)
    promoted = await promote_percepts(UID)
    assert promoted["promoted"] >= 1


async def test_export_roundtrip():
    from medhas.engine import export_user_memory, import_user_memory
    await engine.remember(UID, "Exportable fact about the user", source="user")
    bundle = await export_user_memory(UID)
    assert bundle["user_id"] == UID
    assert len(bundle["tables"].get("atomic_facts", [])) >= 1


async def test_set_affect_and_spaced_review():
    from medhas.engine import set_affect, schedule_review, due_for_review, retention
    from medhas.memory.atomic import insert_fact
    f = await insert_fact(UID, "Affective test fact")
    await set_affect(f.id, 0.6, 0.4)
    async with DatabasePool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT affect_valence, affect_arousal, decay_half_life_days FROM atomic_facts WHERE id=$1;",
            f.id)
    # arousal flattened decay -> half-life should exceed the default 7.0
    assert float(row["affect_valence"]) == 0.6
    assert float(row["affect_arousal"]) == 0.4
    assert float(row["decay_half_life_days"]) > 7.0
    nxt = await schedule_review(f.id, success=True)
    assert nxt is not None


async def test_eval_temporal_consistency():
    facts = [
        {"id": "a", "fact_text": "Lives in Delhi", "valid_from": "2020-01-01T00:00:00",
         "valid_to": "2023-01-01T00:00:00", "contradicted_by": []},
        {"id": "b", "fact_text": "Lives in Hyderabad", "valid_from": "2023-01-02T00:00:00",
         "valid_to": None, "contradicted_by": []},
    ]
    c2022 = temporal_consistency_check(facts, "2022-06-01T00:00:00")
    assert c2022["valid_at_query_time"] == 1
    assert c2022["consistent"] is True
    c2024 = temporal_consistency_check(facts, "2024-06-01T00:00:00")
    assert c2024["valid_at_query_time"] == 1
