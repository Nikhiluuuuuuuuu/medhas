"""Cognition subsystem tests — prove perception, reasoning, generalization, embodiment work.

Runs against the live medhas_test Postgres. ONLINE: every LLM-backed stage (extraction,
resolution, date parsing, dream synthesis) uses the real Groq provider — there is no
offline mode. Each test cleans up its own user_id.
"""

import os
import uuid

import pytest

from medhas.storage import DatabasePool, initialize_schema
from medhas.cognition import perception, reasoning, generalization
from medhas.cognition.embodiment import BodyModel, ensure_body_schema
from medhas.engine import engine as e

PG = os.environ.get("POSTGRES_DB", "medhas_test")


def _uid() -> str:
    return "cog_" + uuid.uuid4().hex[:10]


async def _cleanup(uid: str) -> None:
    async with DatabasePool.acquire() as c:
        for tbl in ("atomic_facts", "graph_nodes", "graph_edges", "episodes", "body_effects"):
            try:
                await c.execute(f"DELETE FROM {tbl} WHERE user_id=$1", uid)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# PERCEPTION
# ---------------------------------------------------------------------------
async def test_perception_extracts_entities_and_relations_offline():
    uid = _uid()
    try:
        p = await perception.perceive("priya launched lumina in 2023 and mentors rahul",
                                       modality="text", user_id=uid)
        assert "priya" in p.entities
        assert any(r[1] == "LAUNCHED" for r in p.relations), p.relations
        assert any(r[1] == "MENTORS" for r in p.relations), p.relations
        assert 0.0 <= p.salience <= 1.0
        assert p.scene_type in ("creation", "social")
    finally:
        await _cleanup(uid)


async def test_perception_structured_modality_adapter():
    # register an adapter that turns raw bytes into text features
    perception.register_adapter("image", lambda raw: f"image shows: {raw}")
    p = await perception.perceive(b"a cat on a roof", modality="image")
    assert p.modality == "image"
    assert "cat" in p.raw or "roof" in p.raw


# ---------------------------------------------------------------------------
# REASONING
# ---------------------------------------------------------------------------
def test_forward_chain_derives_transitive_fact():
    facts = [("a", "MENTORS", "b"), ("b", "MENTORS", "c")]
    rules = reasoning.default_rules()
    closure = reasoning.forward_chain(facts, rules)
    assert ("a", "MENTORS", "c") in closure, closure


def test_forward_chain_works_at_implies_affiliated():
    facts = [("niki", "WORKS_AT", "kraionyx")]
    closure = reasoning.forward_chain(facts, reasoning.default_rules())
    assert ("niki", "AFFILIATED_WITH", "kraionyx") in closure, closure


def test_abduce_finds_missing_premise():
    # Single-premise rule: x CREATED y <- x FOUNDED y.
    # Know (a FOUNDED b); observe (a CREATED b) -> derivable, no abduction.
    obs = ("a", "CREATED", "b")
    known = [("a", "FOUNDED", "b")]
    assert reasoning.abduce(obs, reasoning.default_rules(), known) == []

    # Know nothing; to explain (a CREATED b) we must ABDUCE the single premise (a FOUNDED b).
    res = reasoning.abduce(obs, reasoning.default_rules(), [])
    flat = [t for exp in res for t in exp]
    assert ("a", "FOUNDED", "b") in flat, flat


# ---------------------------------------------------------------------------
# GENERALIZATION
# ---------------------------------------------------------------------------
def test_induce_schema_and_apply():
    inst = [
        ("priya", "MENTORS", "rahul"),
        ("priya", "MENTORS", "sam"),
        ("niki", "MENTORS", "jo"),
    ]
    schemas = generalization.induce_schema(inst)
    assert "MENTORS" in schemas
    assert schemas["MENTORS"].confidence() >= 0.34
    preds = generalization.apply_schema(schemas, "carol", "lee")
    assert "MENTORS" in preds, preds


# ---------------------------------------------------------------------------
# EMBODIMENT
# ---------------------------------------------------------------------------
async def test_body_learns_action_effect():
    uid = _uid()
    try:
        await ensure_body_schema()
        body = BodyModel(uid)
        body.register_capability("send_email", lambda ctx, params: f"sent:{params.get('to')}")
        out = await body.act("send_email", "tell the team", {"to": "team@x.com"})
        assert out["executed"] is True
        assert out["success"] is True
        # prediction should now reflect the learned effect
        pred = await body.predict("send_email", "tell the team")
        assert "sent:" in pred, pred
    finally:
        await _cleanup(uid)


# ---------------------------------------------------------------------------
# FULL COGNITIVE PIPELINE (think)
# ---------------------------------------------------------------------------
async def test_engine_think_runs_pipeline_offline():
    uid = _uid()
    try:
        await ensure_body_schema()
        await initialize_schema()
        body = BodyModel(uid)
        body.register_capability("log", lambda ctx, params: "logged")
        res = await e.think("priya launched lumina and mentors rahul",
                             uid, body=body, action="log", action_params={})
        assert res["percept"] is not None
        assert "priya" in res["percept"]["entities"]
        assert "MENTORS" in {tuple(r)[1] for r in res["percept"]["relations"]}
        assert res["action"]["executed"] is True
    finally:
        await _cleanup(uid)


# ---------------------------------------------------------------------------
# OPEN RELATION VOCABULARY (schema evolution) — the relation set is NOT hard-coded
# ---------------------------------------------------------------------------
async def test_relation_vocabulary_grows_not_hardcoded():
    from medhas.cognition.schema_evolution import (
        record_relation, known_relations, seed_default_relations,
    )
    from medhas.cognition.reasoning import default_rules
    uid = _uid()
    try:
        await initialize_schema()
        # 1. Seed boots the vocabulary from the reasoning-rule relations (legitimate
        #    inference axioms, NOT a hard-coded extraction enum). The seed is NOT the
        #    ceiling — it is just a starting point.
        rule_rels = sorted({r.head[1] for r in default_rules() if r.head[1]})
        await seed_default_relations(uid, rule_relations=rule_rels)
        known = await known_relations(uid)
        # Seeded relations are the rule HEADS (e.g. MENTORS, CREATED, AFFILIATED_WITH,
        # BASED_IN, SHIPS) — note WORKS_AT is a rule *premise*, not a head, so it is
        # correctly absent from the seed. Assert on a real seeded head instead.
        assert "MENTORS" in known and "CREATED" in known, known

        # 2. A NOVEL relation (not in the seed) gets recorded when discovered.
        #    This proves the set grows at runtime instead of being frozen.
        assert "ACQUIRED" not in known
        await record_relation(uid, "acquired", source="extracted")
        known2 = await known_relations(uid)
        assert "ACQUIRED" in known2, known2
        # usage count is tracked
        async with DatabasePool.acquire() as c:
            cnt = await c.fetchval(
                "SELECT usage_count FROM relation_types WHERE user_id=$1 AND relation=$2",
                uid, "ACQUIRED",
            )
            assert cnt == 1
            # re-recording bumps the count (schema evolution, not duplicate rows)
            await record_relation(uid, "acquired", source="extracted")
            cnt2 = await c.fetchval(
                "SELECT usage_count FROM relation_types WHERE user_id=$1 AND relation=$2",
                uid, "ACQUIRED",
            )
            assert cnt2 == 2
    finally:
        async with DatabasePool.acquire() as c:
            await c.execute("DELETE FROM relation_types WHERE user_id=$1", uid)
        await _cleanup(uid)


async def test_extract_graph_open_records_relations_online():
    # Online path: the LLM emits relations (no closed vocabulary) and each discovered
    # relation is recorded into the evolving relation_types vocabulary. Proves extraction
    # is open / LLM-driven, not a frozen hard-coded list.
    from medhas.extraction.llm_extract import extract_graph_open
    uid = _uid()
    try:
        await initialize_schema()
        triples, _ = await extract_graph_open(
            "priya founded lumina and works at kraionyx", user_id=uid
        )
        rels = {r for _s, r, _o in triples}
        assert "FOUNDED" in rels and "WORKS_AT" in rels, rels
        from medhas.cognition.schema_evolution import known_relations
        known = await known_relations(uid)
        assert "FOUNDED" in known and "WORKS_AT" in known, known
    finally:
        async with DatabasePool.acquire() as c:
            await c.execute("DELETE FROM relation_types WHERE user_id=$1", uid)
        await _cleanup(uid)
