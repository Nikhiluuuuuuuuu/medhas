# Unified 6-in-1 AI Agent Memory Engine

A production-grade, ultra-low-latency local AI agent memory engine built in **Python 3.11+** that synthesizes the 6 major memory paradigms (Mem0, Cognee, Letta, Zep/Graphiti, HippoRAG, LightRAG) into a single unified architecture:

1. **Layer 1: Base Session Audit Log (Convex)** — Immutable message logging, history retrieval, and session metadata tracking.
2. **Layer 2: Working Memory RAM (Letta)** — Dynamic JSONB prompt RAM blocks (`user_profile`, `scratchpad`, `active_goals`, `persona`, + custom) editable by the agent mid-session via tool calling, plus a cold archival store (`archive_memory` / `recall_archival`).
3. **Layer 3: Atomic Fact Engine (Mem0)** — HNSW vector + BM25 hybrid search, MD5 hash deduplication, LLM decision matrix (ADD/UPDATE/DELETE/NO_CHANGE), and soft-deletion (`is_active = FALSE`).
4. **Layer 4: Bi-Temporal Knowledge Graph (Zep / Graphiti)** — Entity node canonicalization + semantic merge, bi-temporal edge validity (`valid_from`, `valid_to`), and point-in-time state queries.
5. **Layer 5: Knowledge-Graph Ingestion (Cognee)** — Background episode-anchored extraction pipeline with last-K conversation context and entity resolution.
6. **Layer 6: Associative & Dual-Level Retrieval (HippoRAG / LightRAG)** — Spreading-activation / Personalized PageRank boost over the entity graph, plus `naive` / `local` / `global` / `hybrid` retrieval modes.

---

## Technical Stack & Performance Features

- **Database**: PostgreSQL 16 with `pgvector` extension and HNSW indexing (`atomic_facts_vector_idx`).
- **DB Connection Driver**: `asyncpg` C-extension pool for **sub-2ms DB query latency**.
- **LLM Inference**: Groq Python SDK (`groq`) for high-speed inference (`llama-3.3-70b-versatile`).
- **Vector Embeddings**: `fastembed` / `BAAI/bge-base-en-v1.5` local ONNX CPU embeddings (768-dim, <5ms latency).
- **Data Validation**: Pydantic v2 Rust-core models.
- **Workflow & Timing**: Sub-millisecond performance timers (`measure_latency`).

---

## Online-only (requires Groq)

The engine is **online-only**: relation/edge extraction, entity resolution, date parsing,
and dream-cycle consolidation all call the LLM (Groq `llama-3.3-70b-versatile`). There is
**no offline mode** (`MEDHAS_OFFLINE` / `OFFLINE_MODE` have been removed). Relation
extraction is **open / LLM-driven** — the model emits any relation string and each one is
recorded into the evolving `relation_types` vocabulary, so there is no frozen hard-coded
relation list. If an LLM call fails (e.g. network/429), callers degrade gracefully
(store the raw turn as an atomic fact; return empty graph edges) rather than fabricating
hard-coded edges. A valid `GROQ_API_KEY` is required.

```bash
POSTGRES_DB=medhas_test python -m pytest tests/ -q   # runs against live Groq
```

## Cognition subsystem (`agi/cognition/`)

On top of the 6-layer memory engine sits a cognitive loop (`engine.think()`) that adds the
four properties a memory store lacks on its own:

- **Perception** — `perceive()` turns text/structured/image/audio input into a normalized
  `Percept` (entities, relations, salience, scene type) with modality adapters you can extend.
- **Reasoning** — `reasoning.py` is a real Horn-clause engine: forward chaining over the
  knowledge graph (e.g. `works_at → AFFILIATED_WITH`) plus abduction (explain an observation
  by the minimal missing premise). Deterministic, no LLM.
- **Generalization** — `generalization.py` induces typed relation schemas from few examples
  and transfers structure via analogy (`analogy(a, b, graph)` copies `a`'s relation skeleton
  onto `b`).
- **Embodiment** — `embodiment.py`'s `BodyModel` is a learnable action→effect (affordance)
  model: register capabilities, `act()`, and it records predicted-vs-observed outcomes in
  `body_effects` so the agent improves with experience.

Run `python -m pytest tests/test_cognition.py -q` to see each verified against live Postgres.

## Directory & File Structure (Cognee Modular Design)

```text
.
├── config/
│   └── settings.py                 # System configuration & Pydantic BaseSettings
├── core/
│   ├── exceptions.py               # Application Exception Hierarchy
│   └── interfaces.py               # Abstract Base Classes (ABC)
├── infrastructure/
│   ├── db/
│   │   ├── connection.py          # High-performance asyncpg connection pool manager
│   │   └── schema.py              # Automatic DDL runner & vector extension setup
│   └── llm/
│       ├── groq_provider.py       # Groq SDK LLM completion & tool client
│       └── embedding_provider.py  # FastEmbed local vector embedding provider
├── memory/
│   ├── session/                    # Layer 1: Immutable Session Logging (Convex)
│   │   ├── create_session.py
│   │   ├── log_message.py
│   │   └── get_transcript.py
│   ├── working/                    # Layer 2: Dynamic Prompt RAM (Letta)
│   │   ├── get_blocks.py
│   │   ├── update_block.py
│   │   └── render_prompt.py
│   ├── atomic/                     # Layer 3: Vector Fact Lifecycle (Mem0)
│   │   ├── search_facts.py
│   │   ├── insert_fact.py
│   │   └── deactivate_fact.py
│   └── graph/                      # Layer 4: Bi-Temporal Knowledge Graph (Zep/Graphiti)
│       ├── upsert_node.py
│       ├── update_edge.py
│       ├── query_subgraph.py
│       └── query_point_in_time.py
├── pipeline/
│   ├── hot_path.py                 # Sub-10ms prompt assembly & state loader
│   ├── async_extractor.py          # Non-blocking background fact & edge extraction
│   └── agent_graph.py              # Master Unified Memory Engine Orchestrator
├── schemas/                        # Pydantic v2 schemas for all memory layers
├── tools/                          # Agent self-editing & temporal graph query tool handlers
├── utils/                          # Structured logging & latency timers
├── docker-compose.yml              # Local PostgreSQL 16 + pgvector container spec
├── pyproject.toml                  # Python package specification
├── schema.sql                      # SQL DDL script for database & HNSW index creation
└── main.py                         # Production CLI & automated verification runner
```

---

## Quickstart Guide

### 1. Launch Local Database
Ensure Docker Desktop is running, then start the PostgreSQL container:

```bash
docker-compose up -d
```

### 2. Install Python Dependencies

```bash
pip install -e .
# or
pip install asyncpg pgvector pydantic pydantic-settings groq fastembed rich python-dotenv
```

### 3. Set Groq API Key (Optional for API inference)
Create a `.env` file or export `GROQ_API_KEY`:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

### 4. Run Production Verification Suite

```bash
python main.py
```

This will run all 5 multi-turn verification tests demonstrating sub-millisecond connection pooling, vector HNSW search, preference soft-deletion, bi-temporal graph updating, and cross-paradigm reasoning.
