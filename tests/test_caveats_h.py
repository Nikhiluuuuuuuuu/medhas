"""Regression tests for the three remaining caveats (H1-H3).

H1 — temporal: valid_from is derived from text at ingestion (no manual backdating).
H2 — anaphora: 'the company that builds X' / 'he' / 'she' resolve to concrete entities.
H3 — multi-hop: N-hop (3) traversal + promotion of the chain-terminus fact to rank 1.

All tests run against the live PostgreSQL database (POSTGRES_DB=medhas_test) using a
unique user_id that is cleaned up afterwards. No external connections are used.

Uses pytest-asyncio (asyncio_mode=auto) so the event loop is managed by pytest.
"""

import uuid
from datetime import datetime, timezone

import pytest

from infrastructure.db import DatabasePool, initialize_schema
from agi.engine import engine as e
from utils.dates import extract_fact_date
from agi.anaphora import resolve_query_entities, _most_salient_person


@pytest.fixture(scope="function", autouse=True)
async def _db():
    await DatabasePool.initialize()
    await initialize_schema()
    yield
    await DatabasePool.close()


async def _cleanup(uid: str):
    async with DatabasePool.acquire() as conn:
        await conn.execute("DELETE FROM atomic_facts WHERE user_id=$1", uid)
        await conn.execute("DELETE FROM graph_edges WHERE user_id=$1", uid)
        await conn.execute("DELETE FROM graph_nodes WHERE user_id=$1", uid)


# --------------------------------------------------------------- H1: date extraction
def test_h1_extract_year():
    assert extract_fact_date("Nikhil moved to Hyderabad in 2024") == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_h1_extract_iso():
    assert extract_fact_date("Event on 2026-07-31") == datetime(2026, 7, 31, tzinfo=timezone.utc)


def test_h1_extract_none():
    assert extract_fact_date("Nikhil likes concise answers") is None


async def test_h1_valid_from_derived_at_ingestion():
    uid = "h1_" + uuid.uuid4().hex[:8]
    try:
        await e.remember(uid, "Nikhil lived in Bangalore before moving to Hyderabad in 2024")
        async with DatabasePool.acquire() as conn:
            row = await conn.fetchrow("SELECT valid_from FROM atomic_facts WHERE user_id=$1", uid)
            vf = row["valid_from"]
        assert vf is not None
        # Relative-date resolution ("before 2024" -> 2023) is semantically correct;
        # the literal anchor (2024) is also acceptable. Both prove a date was derived.
        assert vf.year in (2023, 2024)
    finally:
        await _cleanup(uid)


async def test_h1_temporal_before_returns_prior_residence():
    uid = "h1t_" + uuid.uuid4().hex[:8]
    try:
        facts = [
            "Nikhil co-founded Kraionyx AI with 4 friends in Hyderabad in 2025",
            "Kraionyx builds KareOS as its flagship product",
            "Nikhil lived in Bangalore before moving to Hyderabad in 2024",
        ]
        await e.remember_batch(uid, facts, delay=1.5)
        r = await e.recall(uid, "where did Nikhil live before Hyderabad", enforce_abstention=True)
        assert r["status"] == "ok"
        assert r["results"]
        assert "bangalore" in r["results"][0]["fact_text"].lower()
    finally:
        await _cleanup(uid)


# --------------------------------------------------------------- H2: anaphora
async def test_h2_company_that_builds_resolution():
    uid = "h2a_" + uuid.uuid4().hex[:8]
    try:
        facts = [
            "Nikhil co-founded Kraionyx AI with 4 friends in Hyderabad in 2025",
            "Kraionyx builds KareOS as its flagship product",
        ]
        await e.remember_batch(uid, facts, delay=1.5)
        resolved = await resolve_query_entities("who co-founded the company that builds KareOS", uid)
        assert "KareOS" in resolved
        assert "Kraionyx AI" in resolved  # anaphora resolved 'the company' -> Kraionyx AI
    finally:
        await _cleanup(uid)


async def test_h2_pronoun_he_resolves_to_person():
    uid = "h2b_" + uuid.uuid4().hex[:8]
    try:
        facts = ["Nikhil prefers concise answers", "Nikhil lives in Hyderabad"]
        await e.remember_batch(uid, facts, delay=1.5)
        person = await _most_salient_person(uid)
        assert person == "Nikhil"
        resolved = await resolve_query_entities("what does he prefer", uid)
        assert "Nikhil" in resolved
    finally:
        await _cleanup(uid)


async def test_h2_pronoun_recall_returns_preference():
    uid = "h2c_" + uuid.uuid4().hex[:8]
    try:
        facts = ["Nikhil prefers concise answers", "Nikhil lives in Hyderabad"]
        await e.remember_batch(uid, facts, delay=1.5)
        r = await e.recall(uid, "what does he prefer", enforce_abstention=True)
        assert r["status"] == "ok"
        assert r["results"]
        # promotion must surface the preference fact; reconsolidation may rephrase
        # ("prefers" -> "prefer"), so match the relation root, not the exact string.
        top = r["results"][0]["fact_text"].lower()
        assert "nikhil" in top and "prefer" in top
    finally:
        await _cleanup(uid)


# --------------------------------------------------------------- H3: multi-hop
async def test_h3_multihop_promotes_answer_to_rank1():
    uid = "h3_" + uuid.uuid4().hex[:8]
    try:
        facts = [
            "Nikhil co-founded Kraionyx AI with 4 friends in Hyderabad in 2025",
            "Kraionyx builds KareOS as its flagship product",
            "Nikhil lived in Bangalore before moving to Hyderabad in 2024",
        ]
        await e.remember_batch(uid, facts, delay=1.5)
        r = await e.recall(uid, "who co-founded the company that builds KareOS", enforce_abstention=True)
        assert r["status"] == "ok"
        assert r["results"]
        top = r["results"][0]["fact_text"].lower()
        # promotion surfaced the co-founder fact; reconsolidation may rephrase
        # ("co-founded" -> "co-found"), so match the relation root, not exact string.
        assert "nikhil" in top and ("co-found" in top or "found" in top)
    finally:
        await _cleanup(uid)


async def test_h3_multihop_graph_traversal_bridges_entities():
    """KareOS -> Kraionyx AI -> Nikhil must be reachable (no disconnected fragments)."""
    uid = "h3g_" + uuid.uuid4().hex[:8]
    try:
        facts = [
            "Nikhil co-founded Kraionyx AI with 4 friends in Hyderabad in 2025",
            "Kraionyx builds KareOS as its flagship product",
        ]
        await e.remember_batch(uid, facts, delay=1.5)
        from agi.multihop import multihop_recall
        from memory.atomic import search_facts
        hits = await search_facts(uid, "KareOS", limit=5)
        extra = await multihop_recall(uid, "who co-founded the company that builds KareOS", hits, max_facts=8)
        texts = [f["fact_text"].lower() for f in extra]
        # the co-founder fact must be collected through the 2-hop chain
        # (reconsolidation may rephrase "co-founded" -> "co-found")
        assert any("nikhil" in t and ("co-found" in t or "found" in t) for t in texts)
    finally:
        await _cleanup(uid)
