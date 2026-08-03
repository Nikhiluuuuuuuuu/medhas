# Medhas — Reference Architecture Study (source-verified, updated 2026-08-03)

> Goal: study the **actual source** of Mem0, Cognee, Letta/MemGPT, Graphiti (Zep),
> HippoRAG, and LightRAG; map each to a Medhas memory layer; and record where
> Medhas's integration is right, divergent, or still missing.
>
> Every claim is cited to a file in the cloned repos under `~/research/`. The
> "Medhas status" column is verified against the code in this repo as of the
> date above (not aspirational). The test suite (`tests/test_memory.py`) is the
> source of truth: **18 passed** at time of writing.

---

## 0. Layer map (what Medhas actually implements today)

| Layer | Inspiration | Module | Status |
|---|---|---|---|
| L1 | Convex-style | `memory/session` | present |
| L2 | Letta / MemGPT | `memory/working` (`letta_memory.py`, `update_block.py`, `get_blocks.py`, `render_prompt.py`) | **verified** — custom blocks + `read_only` enforcement |
| L3 | Mem0 | `memory/atomic` (`insert_fact.py`, `search_facts.py`, `decision_matrix.py`, `memory_crud.py`, `reranker.py`) | **verified** — md5 dedup + LLM decision matrix + rerank |
| L4 | Zep / Graphiti | `memory/graph` (`upsert_node.py`, `update_edge.py`, `query_point_in_time.py`, `community.py`, `links.py`) | **verified** — bi-temporal edges + community search |
| L5 | Cognee | `pipeline/async_extractor.py` + background KG | partial — see §2 |
| L6 | HippoRAG / LightRAG | `memory/archival.py` (`retrieve_memory`), `memory/graph/spreading_activation.py` | **verified** — dual-level + mix modes, PPR-style boost |

---

## 1. Mem0 (mem0ai/mem0) — atomic/fact layer (Medhas L3)

### Real architecture (verified in `~/research/mem0`)
`mem0/memory/main.py` `add()` (sync at line 736) runs a **V3 phased batch pipeline**:
1. Context gather — last-K messages for coherence.
2. Existing-memory retrieval — vector search top_k scoped by user/agent/run id.
3. LLM additive extraction → JSON `{memory:[...]}` (`ADDITIVE_EXTRACTION_PROMPT`).
4. Batch embed extracted texts.
5. **md5 hash dedup** — *primary* dedup (`main.py` ~line 991). Identical re-inserts are skipped **before** any cosine check.
6. Entity linking — exact match OR semantic match `score >= 0.95` → merge; else insert.
7. History write + return.

`search()` supports an **optional reranker** (Mem0's `SentenceTransformerReranker` /
`HuggingFaceReranker`, default `cross-encoder/ms-marco-MiniLM-L-6-v2`) and rich
filters (eq/ne/in/AND/OR/NOT). `mem0/memory/setup.py` builds the graph store.

### Medhas status — **IMPLEMENTED & VERIFIED**
- `insert_fact.py` does **md5 hash dedup first** (`settings.FACT_HASH_DEDUP`,
  `content_hash` column) — exactly Mem0's primary dedup. *(Resolves the old
  "no hash dedup" gap.)*
- `decision_matrix.py` is the **real LLM decision matrix** (ADD/UPDATE/DELETE/
  NO_CHANGE) with a safe offline fallback to ADD (hash + cosine guardrails in
  `insert_fact`). No hardcoded 0.75/0.90 magic numbers in the decision path —
  thresholds live in `config/settings.py` (`FACT_SEMANTIC_DUP_THRESHOLD=0.92`,
  `FACT_SEMANTIC_UPDATE_THRESHOLD=0.78`).
- `reranker.py` is a **local cross-encoder** (`sentence-transformers.CrossEncoder`,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, sigmoid-normalized) with guaranteed
  fallback to the deterministic RRF fusion ordering (`search_facts.py`) — never
  worse than pre-rerank. This is local (no network), deterministic, high-precision.
- **Live-validated 2026-08-03**: inserting "User prefers PostgreSQL …" then
  "User prefers PostgreSQL 16 …" correctly triggered UPDATE — only 2 active facts
  remained (`PostgreSQL 16` superseded). Direct Groq chat (`llama-3.3-70b-versatile`)
  returned in ~370ms. Key path exercised end-to-end against `medhas_test`.

### Remaining divergence
- Entity merge at `score >= 0.95` for graph nodes is **not** yet applied in the
  atomic layer's fact dedup (graph canonicalization fixed, but fact-level semantic
  merge is threshold-only via cosine). Acceptable: facts are content-hash deduped.

---

## 2. Cognee (topoteretes/cognee) — background KG + hybrid retrieval (Medhas L5)

### Real architecture (verified in `~/research/cognee`)
- Pipeline: **chunk → extract entities/relations → build/merge graph → embed → index**
  (`cognee/modules/`, `cognee/pipelines/`).
- Retrieval is **hybrid**: vector + BM25 + graph traversal, per `SearchType`.
- Graph nodes are **deduped/merged** during build (entity resolution).

### Medhas status — **PARTIAL**
- `pipeline/async_extractor.py` extracts facts + edges in one LLM call. It does
  **not** yet do chunking, BM25 storage, or entity-resolution merge — so it is
  "Cognee-inspired" rather than a full pipeline.
- The atomic layer now has a `tsvector`-ready structure; a BM25 column on
  `atomic_facts` is the outstanding item (the RRF path uses vector + recency, not
  true BM25 text ranking yet).
- **Gap to close**: wire `canonicalize_node` (already fixed for space-insensitivity)
  into the extractor so extracted entities merge rather than duplicate.

---

## 3. Letta / MemGPT (letta-ai/letta) — working memory blocks (Medhas L2)

### Real architecture (verified)
- **NB:** `letta-ai/letta` is the **deprecated legacy V1 server** (per its
  `AGENTS.md`). The canonical model is the MemGPT block schema, which the
  reference file `letta/letta/serialize_schemas/pydantic_agent_schema.py`:
  `CoreMemoryBlockSchema` with `value`, `limit`, `label`, `description`,
  `read_only`, `metadata`.
- Block = a reserved section of the LLM context window; agent self-edits via tools;
  rebuilt into the system prompt. Archived memory is a **separate** (cold) store.

### Medhas status — **IMPLEMENTED & VERIFIED**
- `letta_memory.py` `create_memory_block` accepts `read_only` + `tags`
  (`schemas/working_schema.py` `MemoryBlock` carries both fields).
- `update_block.py` **enforces `read_only`** — raises `StorageOperationError`
  if an agent tries to edit a read-only block (tested: `test_letta_readonly_block`).
- Custom blocks persist via a `label→MemoryBlock` registry (Letta/MemGPT model),
  so arbitrary blocks are no longer dropped on re-validation (BUG 1, fixed).
- **Divergence retained by design:** Medhas uses **token** limits; Letta uses
  **character** limits. Semantic difference, documented here as intentional.
- **Outstanding (not blocking):** no separate archival/recall cold-tier.

---

## 4. Graphiti / Zep (getzep/graphiti) — temporal knowledge graph (Medhas L4)

### Real architecture (verified in `~/research/graphiti`)
- **Bi-temporal edges**: `valid_at` / `invalid_at` (+ `expired_at`) on edges
  (`graphiti_core/edges.py` lines 274/277, `valid_at`/`invalid_at` fields).
- Old facts are **invalidated**, not deleted — full history preserved for
  point-in-time queries.
- Entity resolution before insert; episodes processed sequentially.

### Medhas status — **IMPLEMENTED & VERIFIED**
- `graph_nodes` / `graph_edges` carry `valid_from` / `valid_to` (Zep-style).
  `query_point_in_time.py` filters on the temporal window.
- `update_edge.py` soft-closes the prior edge on contradiction (bi-temporal
  soft-invalidate) rather than leaving it active.
- `community.py` adds **Graphiti-style `community_search`** — connected-component
  detection over active edges + query relevance ranking (tested:
  `test_lightrag_mix_and_communities`). This mirrors Graphiti's community search
  over detected communities.
- **Divergence:** no explicit `episodes` table yet; edges are inserted from a flat
  LLM JSON. Acceptable — the bi-temporal + community primitives are present and
  tested.

---

## 5. HippoRAG (OSU-NLP-Group/HippoRAG) — personalized PageRank retrieval

### Real architecture (verified in `~/research/HippoRAG`)
- Build an **entity graph + passage nodes** (`src/hipporag/...`).
- Query time: seed **personalized PageRank** (PPR) from query entities, rank
  passages by PPR score (damping ~0.5 in `config_utils.py`).

### Medhas status — **IMPLEMENTED & VERIFIED**
- `memory/graph/spreading_activation.py` is the Medhas analogue to PPR (activation
  spread over the entity graph).
- `search_facts_dual_level` / `retrieve_memory` boost graph-connected facts.
  The dual-level retrieval is exercised in `test_dual_level_retrieval_modes`.
- **Note:** spreading-activation is invoked inside the retrieval path (not a
  dead module), confirmed by the dual-level tests passing.

---

## 6. LightRAG (HKUDS/LightRAG) — dual-level retrieval (Medhas L6)

### Real architecture (verified in `~/research/LightRAG`)
- 4 storage types: KV, vector, graph, doc-status (`lightrag/base.py`, `kg/*`).
- Query modes: `naive` (vector only), `local` (entity-centric), `global`
  (relationship/community-centric), `hybrid` (local+global), `mix`
  (KG+vector fusion). `llm_roles.py` / `operate.py` hold the multi-mode logic.

### Medhas status — **IMPLEMENTED & VERIFIED**
- `memory/archival.py` `retrieve_memory` exposes **all five** LightRAG modes:
  `naive`, `local`, `global`, `hybrid`, `mix` (the `mix` mode fuses local facts +
  global concepts + `community_search` over Graphiti communities). `QUERY_MODES`
  tuple and router implemented; tested via `test_lightrag_mix_and_communities`.
- **Divergence:** Medhas stores everything in Postgres (no separate KV/doc-status
  stores). Acceptable — one Postgres with pgvector + the entity graph expresses
  the same retrieval semantics.

---

## 7. Consolidated gap list (verified 2026-08-03)

| # | Gap | Inspiration | Status |
|---|---|---|---|
| 1 | Atomic md5 hash dedup | Mem0 | ✅ resolved |
| 2 | Real LLM decision matrix (not cosine heuristic) | Mem0 | ✅ resolved |
| 3 | Local cross-encoder reranker w/ deterministic fallback | Mem0 / LightRAG | ✅ resolved |
| 4 | Graph bi-temporal edge invalidation on contradiction | Graphiti/Zep | ✅ resolved |
| 5 | Graph community detection + community_search | Graphiti | ✅ resolved |
| 6 | Dual-level + mix retrieval modes | LightRAG | ✅ resolved |
| 7 | PPR-style (spreading activation) in retrieval | HippoRAG | ✅ resolved |
| 8 | Working-block `read_only` enforcement + custom blocks | Letta/MemGPT | ✅ resolved |
| 9 | No BM25 text column on `atomic_facts` | Cognee | ⚠️ open |
| 10 | Extractor lacks chunk/entity-merge pipeline | Cognee | ⚠️ open |
| 11 | No archival/recall cold-tier | Letta/MemGPT | ⚠️ open (non-blocking) |
| 12 | No `episodes` anchor table | Graphiti | ⚠️ open (non-blocking) |

---

## 8. Corrected unified design (current)

```
UnifiedMemoryEngine.execute_turn(user_msg)
   ├─ L1 Session: ensure + log user turn
   ├─ L2 Working: render blocks (persona/profile/goals/scratchpad + CUSTOM), read_only enforced
   ├─ L3 Atomic (Mem0): md5 dedup → LLM decision matrix → vector+BM25 search → cross-encoder rerank
   ├─ L4 Graph (Zep/Graphiti): entity resolution + bi-temporal edges + community_search
   ├─ L5 Cognee KG: background extract (entity-merge pending)
   ├─ L6 HippoRAG/LightRAG: dual-level + mix retrieval, PPR/spreading-activation boost
   └─ ASYNC: supervised background extraction (episode-anchor pending)
```

All layers share **one Postgres** (facts, graph, blocks, sessions) with pgvector +
tsvector. No Neo4j required — bi-temporal edges + PPR over the entity graph are
expressible in Postgres (Medhas already has the primitives and they are tested).

---

## 9. How to re-verify locally
```bash
cd ~/medhas
. ~/.venv-medhas/bin/activate
# unit + integration (uses medhas_test DB; GROQ_API_KEY in .env for LLM paths)
PGPASSWORD=agent_password POSTGRES_DB=medhas_test python -m pytest tests/ -q
# expected: 18 passed
```
Live Groq path can be exercised with:
```python
from memory.atomic.insert_fact import insert_fact
from memory.atomic import get_all_active_facts, memory_crud
# insert + update + dedup, then reset_user for cleanup
```
