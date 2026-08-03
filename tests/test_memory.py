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


# --- NEW: Mem0-style primary hash dedup (Step 1) ---
@pytest.mark.asyncio
async def test_atomic_md5_hash_dedup(user_id):
    a = await insert_fact(user_id, "User lives in Hyderabad")
    b = await insert_fact(user_id, "User lives in Hyderabad")  # identical text -> same hash
    active = await get_all_active_facts(user_id)
    # Only one active fact should exist (hash dedup, independent of cosine).
    assert len(active) == 1, "MD5 hash dedup must collapse identical facts"
    assert a.id == b.id


# --- NEW: semantic node merge (>=0.95) on top of canonicalization (Step 2) ---
@pytest.mark.asyncio
async def test_graph_semantic_entity_merge(user_id):
    from memory.graph.upsert_node import _semantic_match_node
    # Insert a node, then verify the semantic matcher returns the SAME canonical name
    # for an identical re-insert (cosine = 1.0 >= 0.95). This proves the merge path fires.
    a = await upsert_node(user_id, "Kraionyx AI", "Company")
    match = await _semantic_match_node(user_id, "Kraionyx AI")
    assert match == a.name, "identical entity name must semantically merge to the same node"
    b = await upsert_node(user_id, "Kraionyx AI", "Organization")
    assert a.name == b.name, "re-inserting the same entity must resolve to one node"


# --- NEW: edge invalidation on contradiction (Step 3) ---
@pytest.mark.asyncio
async def test_graph_edge_invalidation_on_contradiction(user_id):
    from memory.graph import get_active_edges
    s = await upsert_node(user_id, "ProjectX", "Project")
    t1 = await upsert_node(user_id, "TeamA", "Team")
    t2 = await upsert_node(user_id, "TeamB", "Team")
    e1 = await update_edge(user_id, s.id, t1.id, "owned_by")
    # Contradiction: now owned by TeamB -> e1 must be soft-closed (valid_to set)
    e2 = await update_edge(user_id, s.id, t2.id, "owned_by")
    edges = await get_active_edges(user_id, s.id)
    active_targets = {e["target_id"] for e in edges}
    assert t1.id not in active_targets, "old 'owned_by TeamA' edge must be invalidated"
    assert t2.id in active_targets, "new 'owned_by TeamB' edge must be active"
    # Prior edge must be soft-closed in the DB (bi-temporal). Verify by re-reading its row.
    from infrastructure.db import DatabasePool
    async with DatabasePool.acquire() as conn:
        row = await conn.fetchrow("SELECT valid_to FROM graph_edges WHERE id=$1", e1.id)
        assert row["valid_to"] is not None, "prior edge should have valid_to set (bi-temporal)"


# --- NEW: Letta archival cold store (Step 6) ---
@pytest.mark.asyncio
async def test_archival_recall(user_id):
    from memory.archival import archive_memory, recall_archival
    await archive_memory(user_id, "The deployment runbook lives in internal wiki section 7")
    rows = await recall_archival(user_id, "deployment runbook")
    assert len(rows) >= 1
    assert "runbook" in rows[0]["content"].lower()


# --- NEW: LightRAG dual-level retrieval modes (Step 6) ---
@pytest.mark.asyncio
async def test_dual_level_retrieval_modes(user_id):
    from memory.archival import retrieve_memory
    await insert_fact(user_id, "User prefers Rust for backend services")
    hybrid = await retrieve_memory(user_id, "backend language preference", mode="hybrid")
    assert hybrid["mode"] == "hybrid"
    assert len(hybrid["facts"]) >= 1
    naive = await retrieve_memory(user_id, "backend language preference", mode="naive")
    assert naive["mode"] == "naive"
    assert len(naive["facts"]) >= 1


# --- NEW: deterministic fusion rerank (closes Mem0 rerank gap) ---
@pytest.mark.asyncio
async def test_rerank_puts_exact_match_first(user_id):
    await insert_fact(user_id, "User deploys services with Kubernetes on GKE")
    await insert_fact(user_id, "User likes hiking on weekends")
    res = await search_facts(user_id, "Kubernetes deployment GKE")
    assert len(res) >= 2
    # The fact that directly mentions the query entities must rank first after rerank.
    assert "Kubernetes" in res[0].fact_text, "rerank should surface the exact-match fact first"
    # relevance scores must be in descending order
    scores = [r.rrf_score for r in res]
    assert scores == sorted(scores, reverse=True), "rerank scores must be monotonically descending"
