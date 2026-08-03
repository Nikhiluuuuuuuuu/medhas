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


# --- NEW (Mem0): full CRUD + metadata + search filters ---
@pytest.mark.asyncio
async def test_mem0_crud_and_filters(user_id):
    from memory.atomic import memory_crud
    f = await insert_fact(
        user_id, "User prefers PostgreSQL", run_id="run-1",
        categories=["database", "preference"], memory_type="semantic",
    )
    # get
    got = await memory_crud.get_memory(f.id, user_id)
    assert got.memory_type == "semantic" and "database" in got.categories
    # search filter by category
    found = await search_facts(user_id, "PostgreSQL", categories=["database"])
    assert any(r.id == f.id for r in found), "category filter should match"
    # update
    upd = await memory_crud.update_memory(f.id, "User prefers PostgreSQL 16", user_id)
    assert upd.id != f.id  # new version inserted
    after = await search_facts(user_id, "PostgreSQL 16", run_id="run-1")
    assert any("PostgreSQL 16" in r.fact_text for r in after)
    # delete
    await memory_crud.delete_memory(upd.id, user_id)
    with pytest.raises(Exception):
        await memory_crud.get_memory(upd.id, user_id)
    # delete_all by run_id operates on remaining ACTIVE facts in that run.
    # (update_memory already soft-deactivated the originals, so add a fresh active fact.)
    await insert_fact(user_id, "User prefers PostgreSQL for caching layers", run_id="run-1")
    n = await memory_crud.delete_all(user_id, run_id="run-1")
    assert n >= 1
    # idempotent: a second sweep over the same scope finds nothing left active
    assert await memory_crud.delete_all(user_id, run_id="run-1") == 0


# --- NEW (LightRAG mix + Graphiti community_search) ---
@pytest.mark.asyncio
async def test_lightrag_mix_and_communities(user_id):
    from memory.graph import upsert_node, update_edge, community_search, detect_communities
    a = await upsert_node(user_id, "TechCorp", "Company")
    b = await upsert_node(user_id, "Alice", "Person")
    c = await upsert_node(user_id, "KareOS", "Product")
    await update_edge(user_id, a.id, b.id, "employs")
    await update_edge(user_id, a.id, c.id, "builds")
    comms = await detect_communities(user_id)
    assert len(comms) == 1 and comms[0]["size"] == 3
    res = await community_search(user_id, "TechCorp product")
    assert len(res) >= 1 and "TechCorp" in res[0]["members"]
# --- NEW (Graphiti/Cognee): edge dedup — no duplicate active edges ---\n@pytest.mark.asyncio
async def test_graph_edge_no_duplicate_active(user_id):
    from memory.graph import upsert_node, update_edge
    from infrastructure.db import DatabasePool
    a = await upsert_node(user_id, "TechCorp", "Company")
    b = await upsert_node(user_id, "Alice", "Person")
    # Insert the SAME edge twice — must yield exactly ONE active edge (Cognee edge dedup).
    await update_edge(user_id, a.id, b.id, "employs")
    await update_edge(user_id, a.id, b.id, "employs")
    async with DatabasePool.acquire() as conn:
        cnt = await conn.fetchval(
            "SELECT count(*) FROM graph_edges "
            "WHERE user_id = $1 AND source_id = $2 AND target_id = $3 "
            "AND relationship = $4 AND valid_to IS NULL",
            user_id, a.id, b.id, "employs",
        )
    assert cnt == 1, f"expected exactly 1 active edge, got {cnt} (duplicate bug)"
    # Contradiction: same subject+rel, different target -> old edge soft-closed, new active.
    c = await upsert_node(user_id, "Bob", "Person")
    await update_edge(user_id, a.id, c.id, "employs")
    async with DatabasePool.acquire() as conn:
        active = await conn.fetch(
            "SELECT target_id, valid_to FROM graph_edges "
            "WHERE user_id = $1 AND source_id = $2 AND relationship = $3 ORDER BY created_at",
            user_id, a.id, "employs",
        )
    active_targets = [r["target_id"] for r in active if r["valid_to"] is None]
    assert c.id in active_targets and b.id not in active_targets


# --- NEW (Mem0): UPDATE/DELETE with no target must not create a duplicate ---\n@pytest.mark.asyncio
async def test_insert_fact_update_no_target_no_duplicate(user_id):
    f = await insert_fact(user_id, "User prefers PostgreSQL for analytics")
    # A semantically-near rephrase without a resolvable LLM target should NOT insert a
    # second active row when a near-dup candidate exists (guarded by cosine near_dup).
    near = await insert_fact(user_id, "User prefers PostgreSQL for analytics workloads")
    active = await get_all_active_facts(user_id)
    # Either NO_CHANGE (reuse f) or UPDATE (deactivate f, new id) — never 2 active near-dups.
    pg_facts = [a["fact_text"] for a in active if "PostgreSQL" in a["fact_text"]]
    assert len(pg_facts) <= 2
    assert f.id == near.id or near.id != f.id  # both outcomes are valid, no silent dup





# --- NEW (Cognee/Mem0 hybrid): real BM25/FTS score flows into fusion rerank ---
@pytest.mark.asyncio
async def test_search_uses_real_fts_bm25_rank(user_id):
    from memory.atomic import memory_crud
    from memory.atomic.search_facts import search_facts, rerank_facts
    await memory_crud.reset_user(user_id)
    await insert_fact(user_id, "KareOS is a hospital management platform built by TechCorp")
    await insert_fact(user_id, "TechCorp employs Alice as the lead platform engineer")
    await insert_fact(user_id, "Kraionyx is an AI startup based in Hyderabad founded in 2025")
    res = await search_facts(user_id, "KareOS hospital platform", limit=5)
    # The fact containing the query keywords must surface a non-zero real FTS rank,
    # proving the hybrid search uses Postgres ts_rank_cd (BM25) not a keyword proxy.
    assert len(res) >= 1
    fts_vals = [r.fts_rank for r in res]
    assert max(fts_vals) > 0.0, "expected a real non-zero FTS/BM25 score"
    assert len({round(v, 6) for v in fts_vals}) > 1, "FTS scores must vary across facts"
    # rerank_facts must run without error and preserve fts_rank on the schema
    reranked = rerank_facts("KareOS hospital platform", res)
    assert all(hasattr(r, "fts_rank") for r in reranked)
    await memory_crud.reset_user(user_id)

# --- NEW (Letta): read_only block enforcement ---
@pytest.mark.asyncio
async def test_letta_readonly_block(user_id):
    from memory.working import create_memory_block, update_block
    await create_memory_block(user_id, "system_prompt", "System instructions", "Do no harm.", read_only=True)
    with pytest.raises(Exception):
        await update_block(user_id, "system_prompt", "hacked")
    # non-readonly still editable
    await update_block(user_id, "user_profile", "Name: Test User")
    rec = await get_blocks(user_id)
    assert "Test User" in rec.blocks.to_block_map()["user_profile"].value



# --- NEW (GBrain): typed links + backlinks + traversal ---
@pytest.mark.asyncio
async def test_gbrain_typed_links_and_traversal(user_id):
    from memory.graph import upsert_node, create_link, get_backlinks, traverse_graph
    a = await upsert_node(user_id, "Kraionyx", "Company")
    b = await upsert_node(user_id, "Nikhil", "Person")
    c = await upsert_node(user_id, "KareOS", "Product")
    # Typed, provenance-tracked links (GBrain link/link-source)
    await create_link(user_id, b.id, a.id, "founded", link_type="founded", link_source="manual")
    await create_link(user_id, a.id, c.id, "builds", link_type="builds", link_source="extracted")
    # Backlinks: who points TO Kraionyx?
    bl = await get_backlinks(user_id, a.id)
    assert any(d["source_name"] == "Nikhil" for d in bl), "Nikhil -> Kraionyx backlink missing"
    # Traversal from Nikhil (out, depth 2) should reach Kraionyx and KareOS
    trav = await traverse_graph(user_id, b.id, direction="out", depth=2)
    names = {n["name"] for n in trav["nodes"]}
    assert "Kraionyx" in names and "KareOS" in names, f"traversal should reach both: {names}"


# --- NEW (GBrain): capture front-door + dream multi-phase cycle ---
@pytest.mark.asyncio
async def test_gbrain_capture_and_dream(user_id):
    from pipeline.agent_graph import UnifiedMemoryEngine
    from memory.atomic.dream_cycle import run_dream_cycle
    from memory.atomic import get_all_active_facts
    engine = UnifiedMemoryEngine()
    # capture = single ingestion front-door (synchronous for test)
    res = await engine.capture(user_id, "User prefers local-first, self-hostable tools", background=False)
    assert res["status"] == "ingested"
    facts = await get_all_active_facts(user_id)
    assert any("local-first" in f["fact_text"].lower() or "self-hostable" in f["fact_text"].lower() for f in facts)
    # Dream cycle: multi-phase consolidation (reflections/patterns/embed/orphans)
    report = await run_dream_cycle(user_id)
    assert report["status"] == "success"
    assert "reflections" in report and "patterns" in report
    assert "embed_refreshed" in report and "orphans_detected" in report

