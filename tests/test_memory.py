"""Regression + happy-path tests for the Medhas memory engine.

Run:  pytest -q
Requires a reachable Postgres (medhas_test) with pgvector + uuid-ossp.
"""
import pytest

from memory.atomic import insert_fact, search_facts, get_all_active_facts, deactivate_fact
from memory.graph import upsert_node, update_edge, run_spreading_activation, update_bayesian_belief
from memory.working import create_memory_block, get_blocks, update_block
from memory.procedural import store_skill_playbook, get_skill_playbook


@pytest.mark.asyncio
async def test_atomic_dedup_and_soft_delete(user_id):
    f1 = await insert_fact(user_id, "User prefers PostgreSQL 16")
    await insert_fact(user_id, "User prefers PostgreSQL 16")  # exact dup
    active = await get_all_active_facts(user_id)
    assert len(active) == 1, "exact duplicate should be deduped"
    await deactivate_fact(f1.id)
    assert len(await get_all_active_facts(user_id)) == 0


@pytest.mark.asyncio
async def test_atomic_search_returns_results(user_id):
    await insert_fact(user_id, "User prefers PostgreSQL 16 for JSONB and pgvector queries")
    res = await search_facts(user_id, "PostgreSQL jsonb vector")
    assert len(res) > 0


# --- BUG 1 regression: custom working-memory block content must persist ---
@pytest.mark.asyncio
async def test_working_memory_custom_block_persists_content(user_id):
    await create_memory_block(user_id, "tech_stack", "Primary stack", "Rust 1.75, PostgreSQL 16, Tokio")
    rec = await get_blocks(user_id)
    blocks = rec.blocks.to_block_map()
    assert "tech_stack" in blocks, "custom block label must be retained"
    assert blocks["tech_stack"].value == "Rust 1.75, PostgreSQL 16, Tokio", \
        "BUG 1: custom block content was dropped on read-back"
    # and via update_block round-trip
    await update_block(user_id, "tech_stack", "Rust 1.80, PostgreSQL 17")
    rec2 = await get_blocks(user_id)
    assert rec2.blocks.to_block_map()["tech_stack"].value == "Rust 1.80, PostgreSQL 17"


# --- BUG 2 regression: graph canonicalization must be space-insensitive ---
@pytest.mark.asyncio
async def test_graph_canonicalization_space_insensitive(user_id):
    n1 = await upsert_node(user_id, "New York", "Location")
    n2 = await upsert_node(user_id, "NewYork", "Location")
    assert n1.name == n2.name, "BUG 2: 'New York' and 'NewYork' should resolve to one node"
    # case + spacing
    a = await upsert_node(user_id, "TechCorp", "Company")
    b = await upsert_node(user_id, "techcorp", "Company")
    assert a.name == b.name


@pytest.mark.asyncio
async def test_graph_bayesian_compounding(user_id):
    from memory.graph import upsert_node
    await upsert_node(user_id, "TechCorp", "Company")
    p1 = await update_bayesian_belief(user_id, "TechCorp", likelihood_evidence=0.85, override_prior=0.50)
    # Second call reads the stored posterior from the first call -> compounds further.
    p2 = await update_bayesian_belief(user_id, "TechCorp", likelihood_evidence=0.85)
    assert p2 > p1, "belief should compound (0.50 -> 0.85 -> higher)"


@pytest.mark.asyncio
async def test_procedural_playbook_store_retrieve(user_id):
    await store_skill_playbook(user_id, "deploy rust microservice", ["docker build", "kubectl apply"])
    pb = await get_skill_playbook(user_id, "deploy rust microservice")
    assert pb is not None and pb["task"] == "deploy rust microservice"


@pytest.mark.asyncio
async def test_graph_edge_and_spreading_activation(user_id):
    n1 = await upsert_node(user_id, "Postgres", "Database")
    n2 = await upsert_node(user_id, "Rust", "Language")
    await update_edge(user_id, n1.id, n2.id, "used_by")
    san = await run_spreading_activation(user_id, ["Postgres"])
    assert len(san) >= 1
