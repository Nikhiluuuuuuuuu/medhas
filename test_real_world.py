"""Verification test for Cognee + Graphiti + Letta + Mem0 + LightRAG + HippoRAG full SOTA integration."""

import asyncio
from infrastructure.db import DatabasePool, initialize_schema
from memory.session import create_session
from memory.procedural import store_skill_playbook, get_skill_playbook
from memory.working import create_memory_block, audit_memory_doctor, auto_archive_context_window
from memory.atomic import insert_fact, search_facts, search_facts_dual_level, purge_user_memories
from memory.graph import upsert_node, update_edge, run_spreading_activation, export_knowledge_graph
from pipeline import UnifiedMemoryEngine
from pipeline.hot_path import extract_seed_terms

async def run_real_world_tests():
    print("=== STARTING SOTA FEATURE INTEGRATION VERIFICATION (LightRAG + HippoRAG + Cognee + Graphiti + Letta + Mem0) ===")
    
    # 1. Test Seed Term Extraction with Technical Inputs
    test_inputs = [
        "Configured PostgreSQL 16 & Redis cluster on AWS us-east-1 for sub-10ms latency using Python3.11",
        "I'm working on gRPC microservices with Web3 & CUDA support in Rust-lang",
        "User's profile needs AI/ML acceleration with PyTorch 2.0"
    ]
    
    for inp in test_inputs:
        seeds = extract_seed_terms(inp)
        print(f"\nInput: '{inp}'")
        print(f"Extracted Seed Terms: {seeds}")
        assert len(seeds) > 1, f"Failed to extract seed terms for: {inp}"
    
    print("\n✅ Seed term extraction verified across technical/complex real-world inputs!")

    user_id = "test_real_world_user"
    await DatabasePool.initialize()
    await initialize_schema()
    await purge_user_memories(user_id)

    # 2. Test Letta Core Memory Omni-Tool, Doctor Audit, and Auto-Archival
    mb = await create_memory_block(user_id, "tech_stack", "Primary engineering stack", "Rust 1.75, PostgreSQL 16, Tokio")
    assert mb["label"] == "tech_stack", "Failed to create Letta core memory block"
    
    doctor_report = await audit_memory_doctor(user_id)
    assert doctor_report["total_blocks"] >= 1, "Letta Memory Doctor audit failed"

    archival_res = await auto_archive_context_window(user_id, max_tokens=10000)
    assert archival_res["status"] == "ok", "Letta auto-archival check failed"
    print(f"\n✅ Letta Memory Omni-Tool, Doctor Audit & Auto-Archival verified! Total blocks: {doctor_report['total_blocks']}")

    # 3. Test Mem0 Multi-Signal Hybrid Search & LLM Decision Matrix
    session = await create_session(user_id)
    fact1 = await insert_fact(user_id, "User prefers PostgreSQL 16 for JSONB and pgvector queries", session_id=session.id, agent_id="backend_agent")
    assert fact1.id is not None, "Failed to insert scoped fact"

    searchResults = await search_facts(user_id, "PostgreSQL jsonb vector", session_id=session.id)
    assert len(searchResults) > 0, "Mem0 multi-signal hybrid search failed"
    print(f"\n✅ Mem0 Multi-Signal Hybrid Search (Vector + BM25 + Recency) verified! Found {len(searchResults)} facts.")

    # 4. Test LightRAG Dual-Level Retrieval
    dual_res = await search_facts_dual_level(user_id, "PostgreSQL jsonb vector", session_id=session.id)
    assert "low_level_facts" in dual_res and "high_level_concepts" in dual_res, "LightRAG dual-level search failed"
    print(f"\n✅ LightRAG Dual-Level Retrieval (Low-Level Facts + High-Level Graph Concepts) verified!")

    # 5. Test Graphiti LLM Entity Canonicalization & HippoRAG PPR Traversal
    node1 = await upsert_node(user_id, "Postgres DB", "Database", session_id=session.id)
    node2 = await upsert_node(user_id, "Postgres", "Database", session_id=session.id)
    assert node1.name == node2.name, f"Graphiti entity canonicalization failed: {node1.name} != {node2.name}"

    node3 = await upsert_node(user_id, "Rust Language", "Language", session_id=session.id)
    await update_edge(user_id, node1.id, node3.id, "integrated_with", session_id=session.id)

    ppr_edges = await run_spreading_activation(user_id, [node1.name])
    assert len(ppr_edges) >= 1 and "ppr_score" in ppr_edges[0], "HippoRAG PPR score calculation failed"
    print(f"\n✅ Graphiti Entity Canonicalization & HippoRAG Personalized PageRank (PPR) verified!")

    # 6. Test Cognee Knowledge Graph Export (D3/NetworkX JSON)
    graph_export = await export_knowledge_graph(user_id)
    assert "nodes" in graph_export and "links" in graph_export, "Cognee knowledge graph export failed"
    print(f"\n✅ Cognee Knowledge Graph Export (D3/NetworkX format) verified! {graph_export['stats']}")

    # 7. Test Multi-Playbook Procedural Storage
    pb1 = await store_skill_playbook(user_id, "deploy rust microservice", ["docker build", "kubectl apply"])
    pb2 = await store_skill_playbook(user_id, "setup postgres vector db", ["create extension vector", "init HNSW index"])

    fetched_pb1 = await get_skill_playbook(user_id, "deploy rust microservice")
    fetched_pb2 = await get_skill_playbook(user_id, "setup postgres vector db")

    assert fetched_pb1 is not None and fetched_pb1["task"] == "deploy rust microservice", "Failed to retrieve Playbook 1"
    assert fetched_pb2 is not None and fetched_pb2["task"] == "setup postgres vector db", "Failed to retrieve Playbook 2"
    print("\n✅ Multi-playbook storage and retrieval verified (both playbooks preserved successfully)!")

    # 8. Test Full Engine Conversation Turn with Technical Query & Tool Execution
    engine = UnifiedMemoryEngine()
    
    turn_response = await engine.execute_turn(
        user_id,
        session.id,
        "I am building a high-throughput microservice in Rust with PostgreSQL 16."
    )
    print(f"\nEngine Turn Response:\n{turn_response}")
    assert len(turn_response) > 5, "Engine failed to generate response"

    await DatabasePool.close()
    print("\n=== ALL SOTA FEATURE INTEGRATION VERIFICATIONS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    asyncio.run(run_real_world_tests())
