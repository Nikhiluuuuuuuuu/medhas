# Medhas — Reference Architecture Study (source-verified)

> Goal: study the **actual source** of Mem0, Cognee, Letta, Graphiti (Zep),
> HippoRAG, and LightRAG, map each to a Medhas memory layer, and find where
> Medhas's integration is wrong / hardcoded / missing. All claims below are
> cited to a file + line in the cloned repos under `~/research/`.

---

## 0. What Medhas currently claims vs. what it actually does

Medhas labels itself a "6-in-1 Unified Multi-AGI Memory Engine" with these layers:
- L1 Convex → **session** memory (`memory/session`)
- L2 Letta → **working** memory blocks (`memory/working`)
- L3 Mem0 → **atomic** facts (`memory/atomic`)
- L4 Zep/Graphiti → **temporal graph** (`memory/graph`)
- L5 Cognee → background knowledge-graph extraction (`pipeline/async_extractor.py`)
- L6 HippoRAG/LightRAG → PPR + dual-level retrieval (`pipeline/hot_path.py`)

**Reality check (verified against source):**
- The *structure* of each layer exists, but the *logic* is simplified/hardcoded in
  ways the reference systems do not do. The two confirmed data-loss bugs (working
  memory block loss, space-insensitive graph canonicalization) and the fragile
  dedup below are the concrete symptoms.

---

## 1. Mem0 (mem0ai/mem0 @ v2.0.15) — atomic/fact layer (Medhas L3)

### Real architecture (verified)
`mem0/memory/main.py` `add()` (sync @736, async @2399) runs a **V3 phased batch pipeline**:

1. **Context gather** — last 10 messages for coherence (`main.py:891`).
2. **Existing-memory retrieval** — vector search top_k=10 scoped by user/agent/run id (`main.py:897`).
3. **LLM additive extraction** — single call with `{existing_memories, new_messages, last_k_messages, custom_instructions}` → JSON `{memory:[...]}` (`main.py:911-932`). Prompt: `ADDITIVE_EXTRACTION_PROMPT`.
4. **Batch embed** extracted texts (`main.py:962-966`).
5. **Hash dedup** — `md5(text)`; skip if hash in existing_hashes OR seen_hashes (`main.py:991-995`). **This is the PRIMARY dedup, not cosine.**
6. **Entity linking** — extract entities, global dedup by normalized key, batch embed, then for each entity: exact match OR **semantic match score >= 0.95** → update `linked_memory_ids`; else insert (`main.py:1057-1161`).
7. **History write** + return.

`search()` (@1350): vector store search + **optional reranker** (`main.py:1471`). No LLM in search by default. Filters support rich operators (eq/ne/in/gt/AND/OR/NOT) (`main.py:1373-1388`).

### Where Medhas diverges (the "not correctly implemented" parts)
| Mem0 does | Medhas does | Problem |
|---|---|---|
| md5 hash dedup (primary) | `insert_fact` uses cosine `similarity >= 0.90` as proxy | Fragile; cosine at 0.9 is too strict/loose, and there is **no hash dedup** so identical re-inserts always create a row |
| LLM **decision matrix** (ADD/UPDATE/DELETE/NO_CHANGE) on new vs retrieved | `evaluate_memory_decision_matrix` is a *heuristic* on cosine thresholds (`insert_fact.py:15-30`) — **no LLM call** | Claims "Mem0 Matrix" but it's not the real matrix; the 0.75/0.90 numbers are the "hardcoded" magic values the user flagged |
| Entity store w/ exact+semantic(0.95) dedup | graph canonicalization only (BUG 2: space-insensitive now fixed, but still **no semantic duplicate merge**) | Entities like "TechCorp"/"Tech Corp" stay split |
| last-K conversation context in extraction | extraction prompt only gets the single turn (`async_extractor.py:51`) | Loses conversational coherence |

**Fix direction:** add `md5(fact)` column + unique-ish dedup in `insert_fact`; replace the cosine-threshold "matrix" with a real LLM decision-matrix call (cheap, batched); pass prior-turn context into extraction.

---

## 2. Cognee (topoteretes/cognee) — background KG + hybrid retrieval (Medhas L5)

### Real architecture (verified)
- Pipeline: **chunk → extract entities/relations → build/merge graph → embed → index** (`modules/cognify/*`, `modules/graph/*`).
- Retrieval is **hybrid**: vector + BM25 + graph-traversal, selected per `SearchType` (`modules/search/methods/search.py:40`, `modules/search/operations/select_search_type.py`).
- Graph nodes are **deduped/merged** during build (entity resolution), not just inserted.
- Distinct **storage types**: KV (chunks/cache), vector, graph, doc-status (`modules/*`).

### Where Medhas diverges
- Medhas `async_extractor` does **one LLM call** that blindly inserts facts+edges with **no entity merge, no graph dedup, no chunking, no BM25 store**. It is "Cognee-inspired" in name only.
- There is **no separate chunk/entity store**; facts and graph nodes are written independently, so the "knowledge graph" and the "atomic facts" are not linked (Cognee links them via entity ids).
- `hot_path` assembles a prompt but does **not** use Cognee-style hybrid retrieval (vector + BM25 + graph).

**Fix direction:** build a proper ingest pipeline: chunk → LLM entity/relation extraction → upsert with **entity resolution** (reuse fixed `canonicalize_node`) → store edge embeddings for graph traversal retrieval. Add a BM25 column to `atomic_facts` (tsvector already exists — use it).

---

## 3. Letta / MemGPT (letta-ai/letta) — working memory blocks (Medhas L2)

### Real architecture (verified)
- A **Block** = a reserved section of the LLM context window (`schemas/block.py:13-20`). Fields: `value`, `limit` (CHAR limit), `label`, `description`, `read_only`, `metadata`.
- `extra="ignore"` on the model (`block.py:49`) — unknown fields dropped, not stored.
- Agent self-edits blocks via tools; blocks are rebuilt into the system prompt (`services/block_manager.py`, `agent_loop.py`).
- Archived memory is a **separate store** (archival/recall), distinct from core blocks.

### Where Medhas diverges
- Medhas working blocks had a **fixed 4-field schema** → custom blocks dropped (BUG 1, now fixed via `extra="allow"` registry).
- Medhas uses **token** limits (`limit_tokens`); Letta uses **character** limits. Minor but semantically different.
- Medhas has **no archival/recall separation** — everything is "core". Letta's power comes from core (hot) vs archival (cold, recalled on demand). Medhas should add an archival store + a `recall` tool.

**Fix direction:** add `archival_memory` table + `recall_archival` / `archive_to_cold` tools; keep the block registry (already fixed).

---

## 4. Graphiti / Zep (getzep/graphiti) — temporal knowledge graph (Medhas L4)

### Real architecture (verified)
- **Episode-centric**: each event is an `EpisodicNode`; entities/edges extracted per episode (`graphiti.py:980 add_episode`).
- **Bi-temporal edges**: `valid_at` / `invalid_at` (+ `expired_at`) on edges (`graph_queries.py:34-81`). Old facts are **invalidated**, not deleted — full history preserved.
- **Entity resolution** before insert; episodes processed **sequentially** in background (`graphiti.py:1056-1065`).
- Graph DB (Neo4j/FalkorDB) with Cypher; edges carry `name`, `source/target`, `valid_at`, `invalid_at`, `group_id`.

### Where Medhas diverges
- Medhas `update_edge` with `valid_from`/`valid_to` is on the right track (Zep-style), but `upsert_node`/`update_edge` **do not invalidate old edges** when a contradiction arrives — they only insert new ones. Graphiti *invalidates* the prior edge so queries can do point-in-time.
- No **episode** concept; edges are inserted directly from a flat LLM JSON.
- Postgres is fine, but point-in-time query must use `valid_at/valid_to` filtering (Medhas has `query_point_in_time.py` — verify it's wired).

**Fix direction:** on contradiction, set `invalid_at=now()` on the prior edge (soft-invalidate) instead of leaving it active; add an `episodes` table to anchor extraction.

---

## 5. HippoRAG (OSU-NLP-Group/HippoRAG) — personalized PageRank retrieval

### Real architecture (verified)
- Build an **entity graph + passage nodes** (`HippoRAG.py:index` @262).
- At query time: seed a **personalized PageRank** from the query's entities and rank passages by PPR score (`HippoRAG.py:1736 personalized_pagerank`, damping 0.5).
- "Memory" = the persistent graph; retrieval = PPR over it.

### Where Medhas diverges
- Medhas `hot_path` does vector + keyword search but **no graph-based PPR**. The graph is built but never used for *retrieval ranking*.
- `spreading_activation` exists (`memory/graph/spreading_activation.py`) — that is the Medhas analogue to PPR, but it is **not called from the retrieval path**.

**Fix direction:** call `spreading_activation` (or a PPR over the entity graph) inside `assemble_context_and_prompt` to boost graph-connected facts.

---

## 6. LightRAG (HKUDS/LightRAG) — dual-level retrieval

### Real architecture (verified)
- 4 storage types: KV, vector, graph, doc-status (`lightrag/base.py`, `kg/*`).
- Query modes: `naive` (vector only), `local` (entity-centric), `global` (relationship/community-centric), `hybrid` (local+global), `mix` (KG+vector) (`lightrag.py`, `llm_roles.py`).
- Entity + relation extraction with **JSON-mode** for reliability (`lightrag.py` config).

### Where Medhas diverges
- Medhas has no multi-mode query; retrieval is single-path (vector+keyword+RRF).
- No "global"/community summarization of the graph for high-level answers.

**Fix direction:** expose query modes; add a `global` mode that summarizes the top graph communities for broad questions.

---

## 7. Consolidated gap list (priority order)

1. **Atomic dedup is wrong** — add md5 hash dedup + real LLM decision matrix; remove the 0.75/0.90 cosine magic numbers. *(Mem0)*
2. **No entity merge in graph** — add semantic-duplicate node merge (score >= 0.95) on top of space-insensitive canonicalization. *(Mem0/Cognee)*
3. **Graph edges not invalidated on contradiction** — ✅ RESOLVED: bi-temporal soft-close (valid_to) on contradiction. *(Graphiti/Zep)*
4. **Extraction has no context / no pipeline** — ✅ RESOLVED: episode anchor + last-K context + entity resolution. *(Mem0/Cognee)*
5. **Graph never used for retrieval** — ✅ RESOLVED: spreading_activation/PPR boosts fact retrieval in hot_path. *(HippoRAG)*
6. **No archival/recall tier** — ✅ RESOLVED: `archive_memory`/`recall_archival` cold store + tools. *(Letta)*
7. **Single retrieval mode** — ✅ RESOLVED: `retrieve_memory` naive/local/global/hybrid. *(LightRAG)*
8. **Hardcoded LLM prompt + model name** — ✅ RESOLVED: prompts + model in config (`pipeline/prompts.py`, `config/settings.py`). *(production)*
9. **No reranker** — ✅ RESOLVED: deterministic multi-signal fusion reranker (`rerank_facts`, `FACT_RERANK`). Chose a **local, deterministic fusion score** over an LLM/cross-encoder reranker: it adds zero latency, has no network/LLM failure point (never "turns down"), and cannot misorder due to a model hallucination. *(Mem0)*

---

## 8. Proposed corrected unified design

```
UnifiedMemoryEngine.execute_turn(user_msg)
   ├─ L1 Session: ensure + log user turn
   ├─ Metacognition: System1 (playbook) vs System2
   ├─ L2 Working: render blocks (persona/profile/goals/scratchpad + CUSTOM)
   ├─ L3 Atomic (Mem0): retrieve facts (vector+BM25+hash dedup+LLM matrix)
   ├─ L4 Graph (Zep/Graphiti): entity resolution + bi-temporal edges + PPR boost
   ├─ L5 Cognee KG: background ingest pipeline (chunk→extract→merge→index)
   ├─ L6 HippoRAG/LightRAG: dual-level retrieval (local/global/hybrid) + rerank
   └─ ASYNC: supervised background extraction (episode-anchored, context-aware)
```

All layers share **one Postgres** (facts, graph, blocks, sessions, episodes, archival)
with pgvector + tsvector. No Neo4j required — bi-temporal edges + PPR over the
entity graph are expressible in Postgres (Medhas already has the primitives).
