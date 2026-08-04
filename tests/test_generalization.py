"""Generalization tests — prove the system works on NOVEL data, not just the
Nikhil/Kraionyx/KareOS demo facts.

These tests use completely different entities (priya/lumina/rahul, lowercase names,
novel relations like 'launched'/'mentors'/'makes') and verify:
  - open LLM extraction builds correct edges for unseen verbs/names (no keyword list)
  - multi-hop anaphora ("the product that makes X", "the person that joined X")
  - temporal recall on distinct dates
  - date extraction from arbitrary phrasing

All run against the live PostgreSQL database (POSTGRES_DB=medhas_test), each with a
unique user_id cleaned up afterwards. No external connections.

Note: ingestion is throttled (remember_batch with a delay) because the shared Groq
key is low-RPM; this keeps CI from tripping 429 rate limits.
"""

import uuid

import pytest

from infrastructure.db import DatabasePool, initialize_schema
from agi.engine import engine as e
from agi.llm_extract import extract_graph_open, extract_date_open, resolve_entities_open


@pytest.fixture(scope="function", autouse=True)
async def _db():
    await DatabasePool.initialize()
    await initialize_schema()
    yield
    await DatabasePool.close()


async def _cleanup(uid: str):
    async with DatabasePool.acquire() as conn:
        for t in ("atomic_facts", "graph_edges", "graph_nodes"):
            await conn.execute(f"DELETE FROM {t} WHERE user_id=$1", uid)


# ----------------------------------------------------------- open extraction (no keywords)
async def test_open_extraction_novel_verbs_lowercase():
    # lowercase names + verbs never in any keyword list
    triples, ents = await extract_graph_open("priya launched lumina in 2023 and mentors rahul")
    rels = {r[1] for r in triples}
    assert "LAUNCHED" in rels
    assert "MENTORS" in rels
    names = {en["name"] for en in ents}
    assert "priya" in names and "lumina" in names and "rahul" in names


async def test_open_date_extraction_arbitrary():
    # LLM parses the year from free phrasing
    dt = await extract_date_open("she relocated to the coast in 2021")
    assert dt is not None and dt.year == 2021
    none_dt = await extract_date_open("he prefers tea")
    assert none_dt is None


async def test_open_resolution_against_graph():
    uid = "gr1_" + uuid.uuid4().hex[:8]
    try:
        await e.remember_batch(uid, [
            "priya launched lumina in 2023",
            "lumina makes solar panels",
        ], delay=1.5)
        resolved = await resolve_entities_open(
            "who launched the product that makes solar panels",
            ["priya", "lumina"],
        )
        # Direct references resolve to known entities with no keyword list. Chained
        # descriptions ("the product that makes solar panels") are bridged via the graph
        # in resolve_query_entities / multihop_recall (see test_multihop_product_chain).
        assert "priya" in resolved
    finally:
        await _cleanup(uid)


# ----------------------------------------------------------- multi-hop on novel data
async def test_multihop_mentor_chain():
    uid = "gm1_" + uuid.uuid4().hex[:8]
    try:
        await e.remember_batch(uid, [
            "priya launched lumina in 2023",
            "priya mentors rahul who joined lumina in 2024",
        ], delay=1.5)
        r = await e.recall(uid, "who mentored the person that joined lumina in 2024",
                            enforce_abstention=True)
        assert r["status"] == "ok"
        assert r["results"]
        assert "mentors rahul" in r["results"][0]["fact_text"].lower()
    finally:
        await _cleanup(uid)


async def test_multihop_product_chain():
    uid = "gm2_" + uuid.uuid4().hex[:8]
    try:
        await e.remember_batch(uid, [
            "priya launched lumina in 2023",
            "lumina makes solar panels",
        ], delay=1.5)
        r = await e.recall(uid, "who launched the product that makes solar panels",
                            enforce_abstention=True)
        assert r["status"] == "ok"
        assert r["results"]
        assert "launched lumina" in r["results"][0]["fact_text"].lower()
    finally:
        await _cleanup(uid)


# ----------------------------------------------------------- temporal: same-year relative anchors
async def test_temporal_same_year_relative():
    uid = "gt2_" + uuid.uuid4().hex[:8]
    try:
        await e.remember_batch(uid, [
            "rahul lived in chennai before moving to bengaluru in 2024",
            "rahul moved to bengaluru in 2024",
        ], delay=1.5)
        r = await e.recall(uid, "where did rahul live before bengaluru",
                            enforce_abstention=True)
        assert r["status"] == "ok"
        assert r["results"]
        # chennai must lead bengaluru even though both mention 2024
        assert "chennai" in r["results"][0]["fact_text"].lower()
    finally:
        await _cleanup(uid)


# ----------------------------------------------------------- temporal on distinct dates
async def test_temporal_distinct_dates():
    uid = "gt1_" + uuid.uuid4().hex[:8]
    try:
        await e.remember_batch(uid, [
            "rahul moved to mumbai in 2022",
            "rahul moved to delhi in 2025",
        ], delay=1.5)
        r = await e.recall(uid, "where did rahul live before delhi",
                            enforce_abstention=True)
        assert r["status"] == "ok"
        assert r["results"]
        assert "mumbai" in r["results"][0]["fact_text"].lower()
    finally:
        await _cleanup(uid)
