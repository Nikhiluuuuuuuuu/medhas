# Production Design Spec: Medhas NEXUS Memory Core

**Date:** 2026-07-30  
**Project Code Name:** Medhas (NEXUS Memory Core v1.0.0-PROD)  
**Status:** Approved for Implementation  
**Architecture Type:** Embedded 3-Tier Bi-Temporal Cognitive Memory Engine  

---

## 1. Executive Summary & Design Goals

Medhas is a production-grade, zero-dependency embedded AGI memory engine designed for autonomous AI agents. It addresses vector RAG limitations ("lost in the middle", high latency, loss of temporal state) by coupling an embedded bi-temporal graph database (**KùzuDB**) with localized fast vector indexing, bidirectional spreading activation, SQLite async Write-Ahead Logging (WAL), and Ebbinghaus sleep consolidation.

### Core Guarantees:
1. **Sub-100ms P99 Retrieval SLA:** Uses in-memory HNSW index for seed retrieval + 2-hop bidirectional graph spreading activation.
2. **Strict Token Overhead:** Formats retrieved sub-graphs within a strict **2,000 token budget per prompt**.
3. **100% Zero External Service Dependency:** Pure embedded stack operating on CPU via ONNX Runtime (`all-MiniLM-L6-v2` + GLiNER) and embedded SQLite WAL (no external Redis required).
4. **Bi-Temporal State & Conflict Resolution:** Full tracking of entity state changes over time (`valid_from`, `valid_to`).

---

## 2. System Architecture & Component Diagram

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                          1. MULTIMODAL INGESTION LAYER                             │
│       [ Raw Text Snippets ]      [ Screenshot / OCR Tags ]      [ Audio / Transcripts ] │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                 2. CROSS-MODAL GROUNDING & CANONICALIZATION LAYER                │
│   • ONNX Sentence Transformers (all-MiniLM-L6-v2) ──► Category-Scoped Fuzzy Match │
│   • GLiNER Zero-Shot Entity Extraction ──► Canonical Node Resolution              │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     3. ASYNCHRONOUS SQLITE WAL QUEUE LAYER                        │
│   • Non-blocking Thread-Safe Enqueue ──► Single-Threaded KùzuDB Commit Worker      │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                    4. HYBRID EMBEDDED STORAGE LAYER                               │
│   • KùzuDB (Bi-Temporal Cypher Graph)  • HNSW / Vector Index  • In-Memory BM25 Index│
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│               5. BIDIRECTIONAL SPREADING ACTIVATION & RRF RETRIEVAL               │
│   • Seed Node Fire (Vector/BM25) ──► 2-Hop Bidirectional Energy Propagation       │
│   • Reciprocal Rank Fusion (RRF Reranking) ──► 2,000 Token Formatted Prompt Block  │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│               6. EBBINGHAUS DECAY & SLEEP CONSOLIDATION ENGINE                    │
│   • Exponential Salience Scrubber ──► Abstract Link Discovery ──► Micro-SLM Synthesis│
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Architecture

### 3.1 Data Model & Schemas

#### A. KùzuDB Cypher Schema
```sql
CREATE NODE TABLE Entity(
    id STRING,                  -- Canonical Name (e.g., "Alice Smith")
    category STRING,            -- Entity Category (e.g., "Person", "Project", "System")
    embedding FLOAT[384],       -- Dense Vector (all-MiniLM-L6-v2)
    created_at DOUBLE,          -- Epoch timestamp of creation
    last_accessed DOUBLE,       -- Epoch timestamp of last retrieval
    access_count INT64,         -- Long-term potentiation access counter
    PRIMARY KEY (id)
);

CREATE REL TABLE CONNECTS(
    FROM Entity TO Entity,
    relation STRING,            -- Predicate (e.g., "MANAGES", "BLOCKED_BY")
    reason STRING,              -- Contextual citation
    salience DOUBLE,            -- Ebbinghaus strength score (0.0 to 1.0)
    weight DOUBLE,              -- Edge propagation weight
    valid_from DOUBLE,          -- Epoch timestamp start
    valid_to DOUBLE,            -- Epoch timestamp end (0.0 = currently valid)
    modality STRING             -- Provenance ("text", "vision", "audio")
);
```

#### B. SQLite WAL Queue Schema (`medhas_wal.db`)
```sql
CREATE TABLE IF NOT EXISTS memory_wal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE,
    timestamp REAL,
    source TEXT,
    relation TEXT,
    target TEXT,
    reason TEXT,
    modality TEXT,
    category_src TEXT,
    category_tgt TEXT,
    status TEXT DEFAULT 'PENDING'
);
```

---

## 4. Algorithmic Specifications

### 4.1 Bidirectional Spreading Activation Algorithm
Energy propagates along both incoming and outgoing edges to avoid directional blindspots:

$$A_i^{(t+1)} = A_i^{(t)} + \sum_{j \in \text{Neighbors}(i)} \left( A_j^{(t)} \cdot W_{ji} \cdot \gamma \right)$$

- $W_{ji} = \text{Salience} \times \text{Weight}$
- $\gamma = 0.75$ (decay factor per hop)
- $\tau = 0.15$ (activation cutoff threshold)
- $k = 2$ (maximum propagation depth)
- Neighbors include both outgoing `(a)->(b)` and incoming `(b)->(a)` relationships.

### 4.2 Category-Scoped Entity Canonicalization
To prevent over-merging ("PostgreSQL" vs "MySQL"):
1. Scope candidate lookup to entities matching `category`.
2. Compute string fuzzy ratio (`RapidFuzz`, threshold > 88%).
3. Compute cosine similarity on 384-dim embeddings via HNSW index (threshold > 0.85).
4. If both candidate category and vector/fuzzy match align, resolve to existing ID; otherwise, insert new Canonical Entity Node.

### 4.3 Reciprocal Rank Fusion (RRF) Reranking
Combines sparse BM25 ranks, vector seed ranks, and spreading activation ranks:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{Vector}, \text{BM25}, \text{Activation}\}} \frac{1}{60 + \text{rank}_m(d)}$$

---

## 5. File Structure & Project Modules

```
medhas/
├── __init__.py
├── storage/
│   ├── __init__.py
│   ├── kuzu_engine.py         # KùzuDB connection & Cypher schema management
│   ├── vector_index.py        # Embedded HNSW / vector seed search index
│   └── sqlite_wal.py          # SQLite WAL queue producer/consumer daemon
├── nlp/
│   ├── __init__.py
│   ├── grounding.py           # Embedder (MiniLM) + GLiNER / ONNX NER pipeline
│   └── canonicalizer.py       # RapidFuzz + category-scoped entity resolution
├── retrieval/
│   ├── __init__.py
│   ├── spreading_activation.py# Bidirectional 2-hop energy propagation
│   └── reranker.py            # RRF reranking & 2,000-token prompt formatting
├── consolidation/
│   ├── __init__.py
│   └── ebbinghaus.py          # Salience decay scrubber & micro-SLM synthesis
├── mcp/
│   ├── __init__.py
│   └── server.py              # FastMCP / MCP SDK server (`nexus_remember`, `nexus_recall`)
└── core.py                    # Unified MedhasMemoryCore orchestrator class
```

---

## 6. Self-Review & Verification Plan

### Verification Checklist:
1. **Schema & Database Creation:** Unit test initializing KùzuDB tables and SQLite WAL.
2. **Bidirectional Spreading Activation Test:** Ingest `(A)-[BLOCKED_BY]->(B)` and verify querying `B` returns `A` via incoming activation.
3. **Entity Resolution Isolation:** Confirm `"PostgreSQL"` and `"MySQL"` remain distinct while `"Alice Smith"` and `"Alice"` merge cleanly.
4. **Token Budget Enforcement:** Verify retrieved sub-graph outputs cut off strictly at $\le 2,000$ tokens.
5. **MCP Server Endpoint Verification:** Verify `nexus_remember` and `nexus_recall` return expected JSON responses.
