# Medhas — Production-Grade AGI Memory Roadmap

> **Goal:** evolve Medhas from a 6-in-1 memory engine (Mem0 + Cognee + Letta/MemGPT +
> Graphiti/Zep + HippoRAG + LightRAG) into a **brain-like AGI memory system** — with
> perception (ingest),短期/working memory, long-term consolidation, sleep-time
> reflection, belief revision, forgetting, and metacognitive retrieval — all production-hardened.
>
> **Status anchor:** 2026-08-03. All 12 alignment gaps are RESOLVED (24 tests pass on
> `main`, live Groq key, no mock data). This document lists the **enhancements only** —
> the work required to reach a production, brain-like AGI memory system. Each item is
> tagged with the current module it touches and the inspiration it maps to.
>
> **How to read:** Items are grouped into Phases (P1–P6). Each item has:
> `Inspired by` · `Current state` · `What to build` · `Where` · `Acceptance test`.

---

## Phase 1 — Ingestion & Extraction Depth (the "perception" stage)

### E1. True document-chunking + structured extraction pipeline (Cognee-style)
- **Inspired by:** Cognee `cognify()` pipeline (`tasks/chunking`, `tasks/extraction`).
- **Current state:** `pipeline/async_extractor.py` does a **single LLM call** over the raw
  turn text; within-batch dedup is done but there is **no chunking, no entity/edge
  separation stage, no re-extraction over chunk boundaries**.
- **What to build:**
  - Split input into overlapping chunks (size + overlap, configurable).
  - Per-chunk entity + edge extraction, then **cross-chunk entity resolution** (merge entities
    that appear in different chunks) before persistence.
  - Emit structured `EntityNode` / `Edge` objects (not just free text), feeding `upsert_node`
    and `update_edge` directly.
- **Where:** `pipeline/async_extractor.py` → new `pipeline/chunking.py`, `pipeline/extraction.py`.
- **Acceptance:** a 5-paragraph doc yields entities merged across chunks (no duplicate nodes);
  regression test on chunk-boundary entity.

### E2. Multi-pass self-reflection extraction (Mem0 `add` + reflection)
- **Inspired by:** Mem0 `add()` runs entity + relationship + then a reflection pass.
- **Current state:** one-shot extraction; reflections only appear later in the dream cycle.
- **What to build:** after fact/edge extraction, run a **second LLM pass** that emits higher-order
  insights ("X implies Y") as `is_reflection` facts at ingest time, not only during sleep.
- **Where:** `pipeline/async_extractor.py`, `memory/atomic/insert_fact.py` (`is_reflection` flag exists).
- **Acceptance:** ingest produces reflection-tagged facts; retrievable via `search_facts`.

### E3. Multimodal memory (images / audio / documents) (LightRAG + Mem0 multimodal)
- **Inspired by:** LightRAG multimodal, Mem0 image/PDF ingest.
- **Current state:** text-only (`FastEmbeddingProvider` 384-dim, `GroqLLMProvider` text).
- **What to build:**
  - Add a CLIP/VLM embedder path; store modality + extracted caption in `atomic_facts.metadata`
    and a `modality` column.
  - Route non-text inputs through captioning → existing text pipeline.
- **Where:** `infrastructure/llm/` (new `MultimodalProvider`), `schema.py`, `atomic_facts` schema.
- **Acceptance:** image → caption → fact stored → retrieved by text query.

---

## Phase 2 — Consolidation & "Sleep-Time" Memory (the hippocampal replay / dreaming stage)

### E4. Scheduled sleep-time consolidation (HippoRAG + human sleep replay)
- **Inspired by:** HippoRAG `graph_prompt` offline consolidation; human memory replay during sleep.
- **Current state:** `memory/atomic/dream_cycle.py` exists (`run_dream_cycle`) but is **manually
  triggered** (`POST /memory/dream`), not scheduled.
- **What to build:**
  - A scheduler (cron/background worker) that runs `run_dream_cycle` per user on idle/off-peak.
  - Add consolidation phases: **reflections → patterns → cross-fact contradiction detection →
    belief strengthening → orphan cleanup** (already partially present — formalize as phases).
- **Where:** new `pipeline/consolidation_scheduler.py`; wire to existing `run_dream_cycle`.
- **Acceptance:** overnight job consolidates a user's day-of facts; no contradictions remain active.

### E5. Contradiction & conflict resolution with uncertainty (Graphiti bi-temporal + Zep)
- **Inspired by:** Graphiti `valid_at`/`invalid_at` soft-close; Zep conflict detection.
- **Current state:** `update_edge` soft-closes on new contradictory edge (identity dedup present);
  facts use `deactivate_fact` on UPDATE/DELETE. **No explicit uncertainty/belief score on facts**,
  no "two sources disagree" surfacing.
- **What to build:**
  - Add `confidence`/`belief_confidence` + `source_count` to `atomic_facts` (mirror `belief_revision.py`
    already on graph nodes).
  - When a contradictory fact arrives, keep both as **active-with-uncertainty** and surface the
    disagreement in retrieval (`uncertain: true`).
- **Where:** `infrastructure/db/schema.py`, `memory/atomic/insert_fact.py`, `search_facts` result.
- **Acceptance:** conflicting facts both retrievable; `uncertain` flag present; belief score updated.

### E6. Belief revision on facts (extend Graphiti-style Bayes to atomic facts)
- **Inspired by:** `memory/graph/belief_revision.py` (odds-form Bayes) — currently **graph nodes only**.
- **Current state:** Bayesian compounding exists for graph nodes; facts have no belief update.
- **What to build:** apply the same incremental posterior update to `atomic_facts.belief_confidence`
  when corroborated/contradicted across turns.
- **Where:** `memory/atomic/insert_fact.py`, `memory/atomic/search_facts.py` (rank by belief).
- **Acceptance:** repeated corroboration raises fact belief score; retrieval re-ranks accordingly.

---

## Phase 3 — Forgetting, Salience & Importance (the "what matters" stage)

### E7. Forgetting & decay (HippoRAG + human forgetting curve)
- **Inspired by:** memory decay; HippoRAG relevance decay.
- **Current state:** facts are deactivated (soft-delete) but **never expire by age/irrelevance**;
  `recency` is used in ranking only.
- **What to build:**
  - Time + access-based decay: `last_accessed_at`, `access_count`; a consolidation job deactivates
    low-salience, stale, rarely-accessed facts (configurable half-life).
  - Protect high-belief / high-importance facts from decay.
- **Where:** `schema.py`, `pipeline/consolidation_scheduler.py`, `search_facts` decay term.
- **Acceptance:** low-salience stale facts auto-deactivate; important ones persist.

### E8. Salience / importance learning (Mem0 + Letta)
- **Inspired by:** Mem0 `importance` score; Letta block salience.
- **Current state:** `FactSearchResult.importance_score` exists but is **static/heuristic**.
- **What to build:** learn importance from access frequency, recency, contradiction events, and an
  LLM salience judge for new facts; store + feed into fusion rerank.
- **Where:** `memory/atomic/insert_fact.py`, `rerank_facts`, `schema.py`.
- **Acceptance:** frequently-accessed facts gain importance; ranked higher over time.

### E9. Episodic → semantic compression (human memory: episodes become gist)
- **Inspired by:** human episodic-to-semantic transition; Graphiti episode summarization.
- **Current state:** `episodes` table stores raw episodes; `dream_cycle` makes reflections but
  **does not compress episodes into semantic summaries**.
- **What to build:** periodic job that summarizes old episodes into semantic "gist" facts and
  marks episodes as `compressed`.
- **Where:** `memory/episodes.py`, `pipeline/consolidation_scheduler.py`.
- **Acceptance:** old episode yields a gist fact; episode flagged `compressed`.

---

## Phase 4 — Metacognition & Retrieval Intelligence (the "executive function" stage)

### E10. Broad metacognitive routing (expand `metacognition.py`)
- **Inspired by:** Mem0 `manage`, Letta self-editing, human metacognition.
- **Current state:** `pipeline/metacognition.py` has a **single** `evaluate_cognitive_mode` function;
  not wired into `UnifiedMemoryEngine.execute_turn` retrieval selection.
- **What to build:**
  - Classify query intent (recall / learn / reflect / plan) and **select retrieval mode**
    (naive/local/global/hybrid/mix) + memory tier (working/atomic/graph/archival) per intent.
  - Decide when to trigger consolidation vs. immediate ingest.
- **Where:** `pipeline/metacognition.py`, `pipeline/agent_graph.py`.
- **Acceptance:** different query types hit different tiers/modes; covered by test.

### E11. Multi-agent self-improvement loop (Mem0 + Letta agents)
- **Inspired by:** Mem0 multi-agent, Letta agent runtime.
- **Current state:** single extractor LLM; dream cycle single-pass reflection.
- **What to build:** a critic/refiner agent that reviews extracted facts/edges for quality,
  resolves ambiguity, and proposes schema fixes — runs during consolidation.
- **Where:** `pipeline/agent_graph.py`, new `pipeline/self_improve.py`.
- **Acceptance:** critic catches + fixes a low-quality extraction in test.

### E12. Uncertainty & provenance surfacing in retrieval
- **Inspired by:** Zep/Graphiti provenance; human "I'm not sure."
- **Current state:** retrieval returns facts/edges; **no provenance or confidence in the response**.
- **What to build:** attach `source_episode_id`, `belief_confidence`, `contradicted_by` to every
  returned memory so the agent can express uncertainty.
- **Where:** `search_facts`, `FactSearchResult`, `retrieve_memory` payload.
- **Acceptance:** retrieved item carries episode + belief + conflict pointers.

---

## Phase 5 — Long-Horizon & Cross-Session Memory (the "lifetime" stage)

### E13. Cross-session identity & continuity (Letta agent state + Mem0 user scoping)
- **Inspired by:** Letta persistent agent blocks; Mem0 user/session scoping.
- **Current state:** `user_id` + `session_id` + `agent_id` scoping exists; working blocks per-user;
  **no cross-session "who is this user over time" continuity layer**.
- **What to build:** a persistent `user_profile` (auto-updated from facts) + long-horizon narrative
  memory that survives session churn.
- **Where:** `memory/working.py`, new `memory/user_profile.py`.
- **Acceptance:** profile updates across sessions; surfaced in recall.

### E14. Temporal & causal reasoning over the graph (Graphiti + HippoRAG)
- **Inspired by:** Graphiti temporal queries; HippoRAG causal paths.
- **Current state:** bi-temporal edges + PPR spreading activation; **no explicit causal/temporal query API**.
- **What to build:** query API for "what was true at time T", "what caused X", "sequence of events"
  over `valid_from`/`valid_to`.
- **Where:** `memory/graph/`, `memory/episodes.py`.
- **Acceptance:** temporal/causal query returns correct snapshot.

### E15. Recall evaluation harness (Recall@K, faithfulness, contradiction rate)
- **Inspired by:** Mem0/Letta eval suites; RAG evaluation (Faithfulness, Answer-Relevancy).
- **Current state:** only functional tests; **no quality/recall metrics**.
- **What to build:** golden-set eval: inject known facts, assert Recall@K, assert no
  contradiction in top-k, assert reflection quality. CI-gated.
- **Where:** `tests/eval_memory.py`, `pytest.ini`.
- **Acceptance:** `pytest tests/eval_memory.py` reports Recall@K + faithfulness scores.

---

## Phase 6 — Production Hardening (the "deployable" stage)

### E16. AuthN/AuthZ + tenant isolation (multi-user SaaS)
- **Inspired by:** Letta App Server auth; Graphiti namespaces (`group_id`).
- **Current state:** `server.py` + `live/server.py` are **open FastAPI** (no auth, no rate-limit,
  `user_id` is a plain string — no tenant enforcement).
- **What to build:**
  - API key / OAuth bearer auth (FastAPI `Depends`).
  - Row-level tenant isolation enforced on every query (`user_id` bound to token).
  - `group_id` namespaces for multi-agent partitioning.
- **Where:** `server.py`, `live/server.py`, `infrastructure/security.py`.
- **Acceptance:** unauthenticated request → 401; user A cannot read user B.

### E17. Rate limiting, quotas & cost guardrails
- **Inspired by:** production LLM cost control (token budgets).
- **Current state:** Groq key used directly; no per-user quota, no token budgeting.
- **What to build:** per-user rate limit (slowapi/Redis), token budget per turn, circuit-breaker on
  provider 429/5xx (the live run already hits Groq 429 — needs graceful backoff + fallback).
- **Where:** `server.py`, `infrastructure/llm/groq_provider.py` (add backoff), new `infrastructure/ratelimit.py`.
- **Acceptance:** exceeding quota → 429 with retry-after; provider outage degrades gracefully.

### E18. Observability, tracing & drift monitoring
- **Inspired by:** Graphiti OpenTelemetry; Mem0 tracing.
- **Current state:** `utils.measure_latency` + logs; **no distributed tracing, no metrics endpoint**.
- **What to build:** OpenTelemetry spans per layer, Prometheus metrics (`/metrics`), alert on
  extraction failure rate / latency p95, memory-growth dashboards.
- **Where:** `utils.py`, `server.py`, new `infrastructure/observability.py`.
- **Acceptance:** `/metrics` scrapable; trace per request across layers.

### E19. Horizontal scaling & multi-backend support
- **Inspired by:** Graphiti multi-driver (Neo4j/FalkorDB); Cognee multi-vector store.
- **Current state:** single Postgres + pgvector; in-process connection pool.
- **What to build:** read-replica / connection pooling (PgBouncer), optional graph-store backend
  abstraction for very large graphs, embedding batching, async queue for extraction (already bg,
  but make it durable with retries).
- **Where:** `infrastructure/db/`, `pipeline/async_extractor.py` (durable queue).
- **Acceptance:** load test 1k users; extraction queue survives restart.

### E20. Backup, point-in-time recovery & data sovereignty
- **Inspired by:** user preference for self-hostable / data-sovereign tools.
- **Current state:** raw Postgres; no documented backup/PITR.
- **What to build:** WAL archival + PITR runbook, encrypted snapshots, export/import of a user's
  full memory graph (facts + edges + episodes + blocks) as portable JSON.
- **Where:** `infrastructure/`, new `scripts/export_memory.py`.
- **Acceptance:** full user memory exported → re-imported into fresh DB identically.

---

## Priority order (recommended execution)

| Rank | Item | Why first |
|------|------|-----------|
| 1 | **E16 AuthN/tenant isolation** | blocks any real deployment; security-critical |
| 2 | **E1 Chunking + structured extraction** | raises extraction quality across all downstream layers |
| 3 | **E4 Scheduled consolidation** | turns dream cycle from manual into a real brain-like sleep stage |
| 4 | **E5/E6 Conflict + belief on facts** | gives memory truthfulness + uncertainty (core to AGI memory) |
| 5 | **E7/E8 Forgetting + salience** | without decay the system grows unbounded and loses signal |
| 6 | **E10 Metacognitive routing** | makes retrieval intelligent per intent |
| 7 | **E15 Eval harness** | you can't claim "production" without measurable recall/faithfulness |
| 8 | **E17 Rate-limit + cost guardrails** | needed before multi-user load |
| 9 | **E13 Cross-session continuity** | the "lifetime memory" property |
| 10 | **E3 Multimodal** | breadth of perception |
| 11 | **E9/E11/E12/E14** | depth: compression, self-improvement, provenance, temporal reasoning |
| 12 | **E18/E19/E20** | scale, observe, recover |

---

## Definition of "done" (AGI-memory acceptance criteria)

- [ ] Ingest handles text + documents + multimodal with chunking + cross-chunk resolution.
- [ ] Memory consolidates automatically on a sleep schedule (reflections, patterns, compression).
- [ ] Conflicting facts coexist with explicit uncertainty + belief scores; retrieval ranks by belief.
- [ ] Forgetting decays low-salience stale memory while protecting important facts.
- [ ] Retrieval is metacognitively routed per intent and surfaces provenance + confidence.
- [ ] Cross-session user continuity persists a lifetime narrative.
- [ ] Eval harness proves Recall@K ≥ target and faithfulness ≥ target on a golden set.
- [ ] API is authenticated, tenant-isolated, rate-limited, observable, and backed up.
- [ ] Horizontally scalable and survives provider outages gracefully.

*This file is the single source of truth for AGI-memory enhancements. Alignment gaps (Mem0/Cognee/
Letta/Graphiti/HippoRAG/LightRAG) are tracked separately in `ARCHITECTURE_REFERENCE.md` and are all RESOLVED.*
