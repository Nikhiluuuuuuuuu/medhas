# Medhas — Production-Grade Human-Like AGI Memory: Research-Grounded Roadmap

> **Purpose:** Take Medhas from a 6-in-1 memory engine (Mem0 + Cognee + Letta/MemGPT +
> Graphiti/Zep + HippoRAG + LightRAG) into a **brain-like AGI memory system** — with the
> full human memory lifecycle: encode → consolidate (sleep) → store (episodic/semantic/
> procedural) → retrieve (recognition + recall) → forget (decay) → self-modify.
>
> **Status anchor:** 2026-08-03. All 12 alignment gaps RESOLVED (24 tests pass on `main`,
> live Groq key, no mock data). This document lists **enhancements only**, now grounded in
> cognitive-science and 2024–2026 AGI-memory research (sources cited inline). **36+ enhancements,
> including the missing memory types (prospective, affective, implicit, meta, sensory) and missing
> processes (admission control, interference/forgetting, security/unlearning) found in review.**
>
> **Design spine — CoALA (Cognitive Architectures for Language Agents, Princeton 2023,
> arxiv 2309.02427):** human-like agent memory needs four types —
> **in-context (working)**, **episodic**, **semantic**, **procedural** — plus a structured
> action space to read/write each. Medhas already touches all four (working blocks, episodes,
> atomic facts, procedural playbooks); this roadmap makes each *biologically faithful* and adds
> the missing **consolidation, forgetting, and metacognition** loops.

---

## Primary research sources (read & cited)

| # | Source | Key mechanism used below |
|---|--------|--------------------------|
| [1] | CoALA — Cognitive Architectures for Language Agents (Princeton, 2023, 2309.02427) | 4 memory types; action space |
| [2] | Generative Agents (Park et al., 2023, 2304.03442) | memory stream + reflection + retrieval by recency/importance/relevance |
| [3] | HippoRAG 2 (2025, 2502.14802) | offline indexing → recognition memory (triple filter) → PPR → QA; neuro-grounded |
| [4] | A-MEM: Agentic Memory (NeurIPS 2025, 2502.12110) | observation → contextual note → reflection link → memory EVOLUTION |
| [5] | SCM: Sleep-Consolidated Memory with Algorithmic Forgetting (2026, 2604.20943) | sleep replay + deliberate forgetting |
| [6] | TOKI: Bitemporal Operator Algebra for Contradiction Resolution (2026, 2606.06240) | created/valid/invalid temporal lattice, contradiction operators |
| [7] | Zep / Graphiti + LongMemEval (2025) | bi-temporal KG; 71% temporal recall vs Mem0 49% |
| [8] | Human memory: reconsolidation (PMC3069643), spacing/testing effect, Ebbinghaus forgetting curve | decay + spaced reinforcement |
| [9] | PM-Bench: Prospective Memory in LLM Agents (2607.12385) | future-intent memory E27 |
| [10] | Emotions in AI / affect in episodic memory (2505.01462; DND-16.7) | affective/flashbulb E28 |
| [11] | Implicit memory in LLM agents (2602.08563) | implicit-memory awareness E30 |
| [12] | Adaptive Memory Admission Control (2603.04549) | write-gate E32 |
| [13] | Survey: Security of LTM in LLM Agents / mnemonic sovereignty (2604.16548); MemMorph (2605.26154) | poisoning/unlearning E34 |
| [14] | Cost & Accuracy of LTM in Multi-Agent (2601.07978) — flat RAG/Mem0 77–81% vs Graphiti/cognee 55–56% | store-choice correction E37 |
| [15] | Catastrophic forgetting (PNAS 1611835114; Nature 07711-7); replay (par.nsf.gov 10350729) | interference/rehearsal E33/E36 |
| [16] | Memory for Autonomous LLM Agents (2603.07670) — Baddeley mapping | working-memory eviction E35 |

---

## 0. Corrected memory taxonomy — what the first draft missed

CoALA's 4 types (in-context / episodic / semantic / procedural) are necessary but **not sufficient** for a brain-like system. Research + the 2026 agent-memory literature show **6 more types** that Medhas currently has *zero* support for. These are added as enhancements E27–E31 below and as first-class `MemoryType` values.

| Type | Human analogue | Why it's needed | Medhas today |
|------|----------------|-----------------|--------------|
| **Prospective** | "remember to do X when Y happens" | agents must carry future intentions through activity (PM-Bench 2026, 2607.12385) | ❌ none (no future-intent store) |
| **Affective / emotional** | flashbulb memories; emotional weighting | high-arousal events get flatter decay + priority recall (2505.01462; DND-16.7) | ❌ none (no affect tag) |
| **Implicit** | skills/state encoded without explicit store | LLM state leaks through outputs even w/o memory module (2602.08563) | ❌ unmodelled |
| **Meta-memory** | "knowing what I know / don't know" | distinct from metacognition (control); enables calibration + abstention | ⚠️ conflated w/ metacognition |
| **Sensory / perceptual buffer** | iconic/echoic pre-attentive stage | first stage before working memory; multimodal raw percept | ⚠️ only final text stored |
| **Semantic (procedural) split** | habits vs facts | already in CoALA; listed for completeness | ✅ partial |

**Processes the first draft missed** (added as E32–E37): admission control (what's *worth* storing), interference / retrieval-induced & catastrophic forgetting (stability–plasticity), memory **security / poisoning / unlearning** ("mnemonic sovereignty", survey 2604.16548), and **working-memory eviction** (Baddeley central executive).



### E1. Formalize the 4 CoALA memory types as first-class stores
- **Inspired by:** [1] CoALA; [2] Generative Agents memory stream.
- **Current state:** `working` (blocks), `episodes`, `atomic_facts`, `procedural` playbooks exist but are **not unified under one typed model**; nothing declares "this is episodic vs semantic."
- **What to build:** a `MemoryType` enum + `memory_type` column on each store; a routing table mapping type→store. Episodic = raw episodes; Semantic = distilled facts/insights; Procedural = playbooks; In-context = working blocks + current turn.
- **Where:** `schemas/`, `memory/__init__.py`, `pipeline/agent_graph.py`.
- **Acceptance:** every stored item has a `memory_type`; retrieval can scope by type.

### E2. Episodic→Semantic gist compression (hippocampal replay → neocortex)
- **Inspired by:** [2] reflection; [3] HippoRAG offline indexing; human episodic-to-semantic transition.
- **Current state:** `dream_cycle` makes reflections/patterns but **does not convert episodes into semantic gist**; episodes stay raw.
- **What to build:** scheduled job summarizes old episodes into semantic "gist" facts, marks episode `compressed`, and links gist→source episode (provenance). This is the brain's "episodic memory becomes general knowledge" path.
- **Where:** `memory/episodes.py`, `pipeline/consolidation_scheduler.py`.
- **Acceptance:** old episode yields a gist fact; episode flagged `compressed`; gist retrievable as semantic memory.

### E3. Procedural memory as executable, self-improving skills
- **Inspired by:** [1] procedural memory; [4] A-MEM evolution.
- **Current state:** `procedural/playbook.py` stores playbooks but **no auto-extraction from successful episodes** and no refinement on failure.
- **What to build:** after a successful task, extract a reusable playbook; on failure, mark it deprecated and propose a revision. Treat playbooks as "muscle memory."
- **Where:** `memory/procedural/`, `pipeline/agent_graph.py`.
- **Acceptance:** a repeated successful pattern auto-creates a playbook; a failing one is flagged.

---

## Phase 2 — Encoding & Perception (the "sensory intake" stage)

### E4. Chunking + cross-chunk entity resolution (Cognee-style perceive)
- **Inspired by:** [3] HippoRAG offline passage→triple extraction; Cognee `cognify`.
- **Current state:** `pipeline/async_extractor.py` = single LLM call, no chunking, no structured entity objects.
- **What to build:** overlapping chunking → per-chunk entity/edge extraction → **cross-chunk resolution** (merge same entity across chunks) → structured `EntityNode`/`Edge` → `upsert_node`/`update_edge`. Mirrors HippoRAG's passage-level OpenIE.
- **Where:** new `pipeline/chunking.py`, `pipeline/extraction.py`.
- **Acceptance:** 5-paragraph doc → entities merged across chunk boundaries (no dup nodes).

### E5. Recognition memory filter (HippoRAG 2 "recognition" stage)
- **Inspired by:** [3] HippoRAG 2 recognition memory — a *quick, cheap* pass that filters the KG to candidate triples before expensive PPR.
- **Current state:** retrieval goes straight to vector + PPR; no cheap recognition pre-filter.
- **What to build:** a two-stage retrieve — (1) cheap lexical/FTS+kNN recognition filter to narrow the graph, (2) PPR/semantic over the narrowed set. This is why HippoRAG 2 beats flat RAG on associative recall.
- **Where:** `memory/graph/spreading_activation.py`, `memory/atomic/search_facts.py`.
- **Acceptance:** recall@k improves on multi-hop queries vs single-stage.

### E6. Multimodal encoding (CLIP/VLM → caption → memory)
- **Inspired by:** [1] unified representation; LightRAG multimodal; Mem0 multimodal.
- **Current state:** text-only (`FastEmbeddingProvider` 384-dim).
- **What to build:** multimodal embedder path; `modality` + `caption` in `atomic_facts.metadata`; non-text → caption → existing text pipeline. Enables "remember this image/sound."
- **Where:** `infrastructure/llm/` (new `MultimodalProvider`), `schema.py`.
- **Acceptance:** image → caption → fact stored → retrieved by text query.

---

## Phase 3 — Consolidation & Sleep (the "hippocampal replay" stage)

### E7. Scheduled sleep-time consolidation (SCM + human sleep replay)
- **Inspired by:** [5] SCM sleep-consolidated memory; [2] reflection loop; human offline consolidation during sleep/quiet rest.
- **Current state:** `run_dream_cycle` exists but is **manual** (`POST /memory/dream`); no scheduler.
- **What to build:** a durable background worker that runs per-user consolidation on idle/off-peak:
  **phases = reflect → detect contradictions → strengthen belief → compress episodes → prune orphans.** This is the single most brain-like addition.
- **Where:** new `pipeline/consolidation_scheduler.py` wrapping `run_dream_cycle`.
- **Acceptance:** overnight job consolidates a user's day; no active contradictions remain.

### E8. Memory evolution on integration (A-MEM observe→note→link→evolve)
- **Inspired by:** [4] A-MEM — each new memory gets a contextual NOTE, is LINKED to related memories, and triggers EVOLUTION of those notes.
- **Current state:** new facts are inserted; existing memories are **not updated** when a related new one arrives (except edge soft-close).
- **What to build:** on insert, (1) write a contextual note, (2) link to top-k similar memories, (3) re-write/extend those memories' notes (evolution). This makes memory a *living graph*, not a log.
- **Where:** `memory/atomic/insert_fact.py`, new `memory/atomic/evolve.py`.
- **Acceptance:** inserting "X moved to Y" updates the note of the prior "X lives at Z" memory.

### E9. Spaced reinforcement & retrieval practice (spacing + testing effect)
- **Inspired by:** [8] spacing effect (distributed practice strengthens memory), testing effect (retrieval > rereading).
- **Current state:** facts have no "review schedule"; reactivation only happens on query.
- **What to build:** a `next_review_at` (Leitner/spaced-repetition schedule) per fact; consolidation job *actively quizzes* (retrieval practice) low-recency important facts to strengthen them — not just passive decay. This is the testing effect applied to agent memory.
- **Where:** `schema.py`, `pipeline/consolidation_scheduler.py`.
- **Acceptance:** important stale facts get scheduled reviews; belief strengthens on successful reactivation.

---

## Phase 4 — Belief, Conflict & Truth (the "what is real" stage)

### E10. Bitemporal contradiction lattice (TOKI + Zep/Graphiti)
- **Inspired by:** [6] TOKI bitemporal operator algebra; [7] Zep/Graphiti `created_at/valid_at/invalid_at`.
- **Current state:** `update_edge` soft-closes contradictory edges; facts use `deactivate_fact`. **No explicit valid-time lattice, no contradiction operators.**
- **What to build:** full bitemporal model on facts + edges with `created_at`, `valid_from`, `valid_to`; operators: `contradicts`, `precedes`, `supersedes`. Query "what was true at T" resolves via the lattice. This is what gives Zep 71% vs Mem0 49% on LongMemEval.
- **Where:** `infrastructure/db/schema.py`, `memory/atomic/insert_fact.py`, `memory/graph/update_edge.py`.
- **Acceptance:** temporal query returns correct snapshot; contradictory facts coexist with `valid_to` set.

### E11. Belief revision on facts (extend Graphiti-style Bayes to atomic facts)
- **Inspired by:** [7] Graphiti belief; existing `memory/graph/belief_revision.py` (graph-only today).
- **Current state:** Bayesian compounding only on graph nodes.
- **What to build:** apply incremental odds-form posterior to `atomic_facts.belief_confidence` on corroboration/contradiction; rank retrieval by belief.
- **Where:** `memory/atomic/insert_fact.py`, `rerank_facts`.
- **Acceptance:** repeated corroboration raises fact belief; retrieval re-ranks.

### E12. Uncertainty & provenance in retrieval ("I'm not sure")
- **Inspired by:** [7] Zep provenance; human metacognitive uncertainty.
- **Current state:** retrieval returns facts/edges; **no belief/conflict/provenance in the payload.**
- **What to build:** attach `source_episode_id`, `belief_confidence`, `contradicted_by[]`, `valid_at/valid_to` to every returned memory so the agent can express calibrated uncertainty.
- **Where:** `FactSearchResult`, `retrieve_memory` payload.
- **Acceptance:** retrieved item carries episode + belief + conflict pointers.

---

## Phase 5 — Forgetting & Pruning (the "healthy forgetting" stage)

### E13. Algorithmic forgetting (SCM + Ebbinghaus curve)
- **Inspired by:** [5] SCM "algorithmic forgetting"; [8] Ebbinghaus forgetting curve.
- **Current state:** facts soft-deleted but **never expire by age/irrelevance**; recency used in ranking only.
- **What to build:** time + access-based decay with a configurable half-life; protect high-belief/high-importance; consolidation job deactivates low-salience stale memories. Matches SCM's deliberate forgetting for signal clarity.
- **Where:** `schema.py`, `pipeline/consolidation_scheduler.py`, `search_facts` decay term.
- **Acceptance:** low-salience stale facts auto-deactivate; important ones persist.

### E14. Salience / importance learning (Mem0 + Letta + Generative Agents)
- **Inspired by:** [2] Generative Agents importance score; Mem0 `importance`; Letta block salience.
- **Current state:** `FactSearchResult.importance_score` exists but is **static/heuristic**.
- **What to build:** learn importance from access freq, recency, contradiction events, and an LLM salience judge at ingest; feed fusion rerank. High-importance facts resist forgetting (E13).
- **Where:** `insert_fact`, `rerank_facts`, `schema.py`.
- **Acceptance:** frequently-accessed facts gain importance; ranked higher; protected from decay.

### E15. Reconsolidation on reactivation (memory becomes labile when recalled)
- **Inspired by:** [8] reconsolidation (PMC3069643) — retrieving a memory makes it malleable; reinforced experience strengthens it.
- **Current state:** retrieval is read-only; no state change on recall.
- **What to build:** on retrieval, mark memory `last_accessed`, bump `access_count`, and (if contradicted during the same turn) trigger reconsolidation update rather than silent dup. This is the biological "recall to update" loop.
- **Where:** `search_facts`, `insert_fact`.
- **Acceptance:** recalling a fact updates its access metadata; contradiction during recall triggers evolution.

---

## Phase 6 — Metacognition & Executive Control (the "prefrontal" stage)

### E16. Intent-aware retrieval routing (executive function)
- **Inspired by:** [1] CoALA action space; [4] A-MEM; human metacognition.
- **Current state:** `pipeline/metacognition.py` has **one** `evaluate_cognitive_mode` fn, unwired into `execute_turn`.
- **What to build:** classify intent (recall/learn/reflect/plan) → select memory tier (working/atomic/graph/archival) + mode (naive/local/global/hybrid/mix) + whether to consolidate now. Wire into `UnifiedMemoryEngine.execute_turn`.
- **Where:** `pipeline/metacognition.py`, `pipeline/agent_graph.py`.
- **Acceptance:** different intents hit different tiers/modes; test-covered.

### E17. Multi-agent self-improvement loop (critic/refiner)
- **Inspired by:** [4] A-MEM evolution; Letta agent runtime; Mem0 multi-agent.
- **Current state:** single extractor LLM; dream cycle single-pass.
- **What to build:** a critic agent reviews extracted facts/edges for quality, resolves ambiguity, proposes schema fixes — runs during consolidation (E7).
- **Where:** `pipeline/agent_graph.py`, new `pipeline/self_improve.py`.
- **Acceptance:** critic catches + fixes a low-quality extraction in test.

### E18. Calibrated confidence & abstention
- **Inspired by:** [7] provenance; human "I don't know."
- **Current state:** no abstention; system returns best-effort.
- **What to build:** if top-k retrieval belief/coverage is below threshold, the agent responds with calibrated uncertainty or "insufficient memory," rather than hallucinating. Ties to E12.
- **Where:** `retrieve_memory`, `pipeline/agent_graph.py`.
- **Acceptance:** low-coverage query yields an explicit uncertainty response.

---

## Phase 7 — Long-Horizon & Continuity (the "lifetime" stage)

### E19. Cross-session identity & lifetime narrative (Letta + Mem0 user scoping)
- **Inspired by:** [1] user-scoped semantic memory; Letta persistent agent blocks.
- **Current state:** `user_id`/`session_id`/`agent_id` scoping exists; **no persistent cross-session "who is this user" layer.**
- **What to build:** auto-updated `user_profile` from facts + long-horizon narrative that survives session churn and is injected as working context.
- **Where:** `memory/working.py`, new `memory/user_profile.py`.
- **Acceptance:** profile updates across sessions; surfaced in recall.

### E20. Temporal & causal query API (Graphiti + HippoRAG)
- **Inspired by:** [7] temporal KG; [3] PPR causal paths.
- **Current state:** bi-temporal edges + PPR; **no explicit temporal/causal query API.**
- **What to build:** API for "what was true at T", "what caused X", "event sequence" over `valid_from/valid_to`.
- **Where:** `memory/graph/`, `memory/episodes.py`.
- **Acceptance:** temporal/causal query returns correct snapshot.

### E21. Recall evaluation harness (Recall@K, faithfulness, LongMemEval-style)
- **Inspired by:** [7] LongMemEval benchmark (temporal recall); RAG faithfulness.
- **Current state:** functional tests only; **no quality metrics.**
- **What to build:** golden-set eval — inject known facts, assert Recall@K, no-contradiction-in-top-k, reflection quality, temporal accuracy. CI-gated.
- **Where:** `tests/eval_memory.py`.
- **Acceptance:** `pytest tests/eval_memory.py` reports Recall@K + faithfulness + temporal accuracy.

---

## Phase 7b — Missing memory types (the gaps from the first draft)

### E27. Prospective memory (future intentions / reminders)
- **Inspired by:** PM-Bench 2026 (2607.12385) — carry a future intention through activity, fire only on cue/time.
- **Current state:** no future-intent store; reminders impossible.
- **What to build:** `prospective_memory` (cue/time, intent, done flag); a lightweight scheduler checks cues each turn and injects the intention into working memory when triggered. Distinct from episodic (past) memory.
- **Where:** `memory/prospective.py`, `pipeline/agent_graph.py` (cue check per turn).
- **Acceptance:** "remind me to X when Y" fires exactly when Y occurs; doesn't fire otherwise.

### E28. Affective / emotional memory (flashbulb weighting)
- **Inspired by:** [2505.01462] affect interwoven with episodic memory; DND-16.7 emotional arousal → flatter decay.
- **Current state:** no affect dimension; all facts decay equally.
- **What to build:** `affect` tag (valence/arousal) on facts/episodes; high-arousal → flatter decay curve + priority in retrieval (flashbulb effect). Enables "this mattered."
- **Where:** `schema.py`, `search_facts` decay + ranking, `async_extractor` (detect affect).
- **Acceptance:** high-arousal fact persists longer + ranks higher; low-arousal decays normally.

### E29. Meta-memory (knowing what you know)
- **Inspired by:** human meta-memory; distinct from metacognition (control). Clarifies E16/E18.
- **Current state:** metacognition and meta-memory conflated.
- **What to build:** a `meta` record per user: coverage stats (how many topics known, confidence per topic), "known unknowns" list. Powers honest "I don't know" and targeted learning.
- **Where:** `memory/metamemory.py`, `retrieve_memory`.
- **Acceptance:** agent reports calibrated coverage; abstains on unknown topics.

### E30. Implicit-memory awareness (state leakage)
- **Inspired by:** [2602.08563] implicit memory — LLM encodes state in outputs; re-ingested outputs become "memory."
- **Current state:** unmodelled; extracted "facts" may actually be the model's own prior leaking.
- **What to build:** tag memories `explicit` vs `implicit-inferred`; down-weight implicitly-inferred items in belief; allow audit of which facts came from user vs model inference.
- **Where:** `insert_fact` (source provenance), `FactSearchResult`.
- **Acceptance:** inferred-vs-stated provenance is queryable.

### E31. Sensory / perceptual buffer (pre-attentive raw percept)
- **Inspired by:** iconic/echoic buffer; first stage before working memory.
- **Current state:** only final structured text stored; raw percept lost.
- **What to build:** short-lived `percept_buffer` (multimodal embedding + raw caption, TTL seconds) feeding encoding; enables "what did I just see" within a turn.
- **Where:** `memory/perceptual.py`, `pipeline/async_extractor.py`.
- **Acceptance:** within-turn raw percept retrievable; auto-expires.

---

## Phase 7c — Missing processes (admission, interference, security, working-memory)

### E32. Adaptive admission control (what's WORTH storing)
- **Inspired by:** Adaptive Memory Admission Control (2603.04549) — gate at write, not only decay at read.
- **Current state:** everything extracted is stored; no write-time gating.
- **What to build:** a scoring gate (novelty × importance × credibility × capacity headroom) decides store / merge / drop. Prevents memory bloat at the source.
- **Where:** `pipeline/async_extractor.py`, `memory/atomic/insert_fact.py`.
- **Acceptance:** low-value duplicate/tautology is dropped at write; important novel fact stored.

### E33. Interference & forgetting mitigation (stability–plasticity)
- **Inspired by:** catastrophic forgetting (PNAS 1611835114; Nature 07711-7); retrieval-induced forgetting.
- **Current state:** decay exists (E13) but **no protection against new learning overwriting old**, no rehearsal buffer.
- **What to build:** (a) protected "core" memories (high belief) are frozen from decay/overwrite; (b) rehearsal buffer replays protected memories during consolidation (stability); (c) detect interference when a new fact would contradict a core memory and route to E10/E11 instead of silent overwrite.
- **Where:** `consolidation_scheduler.py`, `insert_fact`, `schema.py`.
- **Acceptance:** learning new info doesn't erase a protected core fact; contradiction surfaces.

### E34. Memory security — poisoning defense + mnemonic sovereignty
- **Inspired by:** Survey on Security of LTM in LLM Agents (2604.16548, "mnemonic sovereignty"); MemMorph tool-hijacking (2605.26154); DSRM deceptive reasoning.
- **Current state:** **open API (E22) + no poisoning defense + no unlearning.** A malicious turn could inject false facts that later steer the agent.
- **What to build:**
  - **Integrity:** sign/attest memory writes; provenance + source trust score per fact; flag memories from untrusted sources.
  - **Poison detection:** contrastive check — does a new fact contradict high-belief core memory from a *different* trusted source? If yes, quarantine (don't auto-merge).
  - **Unlearning / right-to-be-forgotten:** `forget(user_id, scope)` that cryptographically purges + re-derives dependent reflections; satisfies GDPR-style erasure.
  - **Sandboxing:** memory used for reasoning is separated from memory used for tool calls (prevents MemMorph tool hijack via recalled memory).
- **Where:** `infrastructure/security.py`, `insert_fact`, `memory/atomic/memory_crud.py` (forget), `retrieve_memory` (trust gating).
- **Acceptance:** injected false fact is quarantined, not merged; `forget` fully purges + dependent reflections; tool-call memory isolated from reasoning memory.

### E35. Working-memory eviction policy (Baddeley central executive)
- **Inspired by:** Baddeley model; "Memory for Autonomous LLM Agents" (2603.07670) — LLM=executive, context window=buffer, capacity-limited.
- **Current state:** working blocks are static; no capacity management of the live context window.
- **What to build:** an eviction policy for the active context — keep high-relevance/recent/important, evict low-value, summarise overflow into working blocks. Mirrors the central executive's selectivity.
- **Where:** `memory/working.py`, `pipeline/agent_graph.py`.
- **Acceptance:** overflow turn keeps the most relevant items; nothing critical dropped silently.

### E36. Continual-learning rehearsal buffer (anti-catastrophic-forgetting at scale)
- **Inspired by:** replay methods (par.nsf.gov 10350729); synaptic consolidation analogy.
- **Current state:** no rehearsal of old memories during new learning.
- **What to build:** a bounded rehearsal buffer sampled during consolidation (E7) so old knowledge is periodically reactivated — the agent equivalent of synaptic consolidation.
- **Where:** `consolidation_scheduler.py`.
- **Acceptance:** periodic reactivation of old memories prevents silent loss.

### E37. Process → mechanism decision matrix (choose the BEST one per process)
This answers "for each process, pick the best mechanism" — comparative, not just a list.

| Process | Candidate mechanisms | **Chosen (best-fit)** | Why (evidence) |
|---------|----------------------|------------------------|----------------|
| **Associative / multi-hop retrieval** | dense-only, sparse/BM25, graph+PPR | **Graph + PPR (HippoRAG 2)** + cheap recognition pre-filter | flat RAG misses associations; PPR over KG beats dense on multi-hop; recognition filter cuts cost [3] |
| **Recall of a specific past fact** | vector kNN, FTS/BM25, exact-md5 | **Hybrid: md5/exact → FTS → vector** | exact is cheapest/perfect when available; BM25 for lexical; vector for paraphrase |
| **Temporal "what was true at T"** | versioned rows, bitemporal lattice | **Bitemporal lattice (TOKI/Zep)** | resolves contradictions + valid-time queries; Zep 71% vs Mem0 49% on LongMemEval [6][7] |
| **Fact dedup** | exact hash, semantic cluster, LLM judge | **Hash (fast) → semantic merge (0.95) → LLM only on conflict** | hash is O(1); semantic catches paraphrases; LLM only when needed (cost) |
| **Long-term store** | flat RAG, graph (Graphiti/cognee), full-context | **Hybrid: flat facts (77–81% acc) + graph for relations** | 2601.07978: Mem0/RAG/full-context 77–81% vs Graphiti/cognee 55–56% on accuracy → keep flat store primary, graph as enrichment, not sole source |
| **Belief update** | overwrite, avg, Bayesian odds-form | **Bayesian odds-form (incremental)** | compounding posterior; existing `belief_revision.py` proves it [7] |
| **Forgetting** | TTL delete, Ebbinghaus decay, SCM algorithmic | **Decay (Ebbinghaus) + admission gating + protected core** | SCM shows deliberate forgetting beats pure TTL; gate at write (E32) prevents bloat [5][8] |
| **Consolidation trigger** | manual, scheduled sleep, event-driven | **Scheduled sleep (SCM) + event-driven on contradiction** | sleep replay is brain-faithful; event-driven catches urgent conflicts [5] |
| **Memory write gate** | store-all, LLM-judge, scored admission | **Scored admission (novelty×importance×credibility)** | 2603.04549 shows admission control cuts bloat without losing signal |
| **Security** | none, trust-score, signed+quarantine | **Signed writes + source trust + quarantine + unlearn** | 2604.16548: security can't be retrofitted at retrieve time; mnemonic sovereignty needs write-time integrity |

> **Key correction from research:** the 2026 accuracy study (2601.07978) shows **graph-only memory (Graphiti/cognee) underperforms flat RAG/Mem0 (55–56% vs 77–81%)**. So Medhas must keep its **flat atomic-fact store as the primary recall path** and use the **graph as a relational enrichment layer**, not the sole source. The roadmap's Phase 2/4 graph work is reframed as *enrichment*, not replacement.

---

## Phase 8 — Production Hardening (the "deployable" stage)

### E22. AuthN/AuthZ + tenant isolation
- **Inspired by:** Letta App Server auth; Graphiti `group_id` namespaces.
- **Current state:** `server.py`/`live/server.py` are **open FastAPI** (no auth, no rate-limit).
- **What to build:** API-key/OAuth bearer (`Depends`); row-level tenant isolation; `group_id` partitioning.
- **Where:** `server.py`, `infrastructure/security.py`.
- **Acceptance:** unauth → 401; user A cannot read user B.

### E23. Rate limiting, quotas & cost guardrails
- **Inspired by:** production LLM cost control; live run already hits Groq 429.
- **Current state:** Groq key used directly; no quota; no backoff.
- **What to build:** per-user rate limit (Redis/slowapi), token budget/turn, circuit-breaker + graceful fallback on 429/5xx.
- **Where:** `server.py`, `groq_provider.py`, new `infrastructure/ratelimit.py`.
- **Acceptance:** over-quota → 429 retry-after; provider outage degrades gracefully.

### E24. Observability, tracing & drift monitoring
- **Inspired by:** [7] Graphiti OpenTelemetry; Mem0 tracing.
- **Current state:** `utils.measure_latency` + logs; no tracing/metrics endpoint.
- **What to build:** OTel spans per layer, Prometheus `/metrics`, alerts on extraction-fail/latency p95, memory-growth dashboards.
- **Where:** `utils.py`, `server.py`, new `infrastructure/observability.py`.
- **Acceptance:** `/metrics` scrapable; per-request trace.

### E25. Horizontal scaling & multi-backend
- **Inspired by:** [7] Graphiti multi-driver (Neo4j/FalkorDB); Cognee multi-vector store.
- **Current state:** single Postgres + pgvector; in-process pool.
- **What to build:** PgBouncer/pooling, optional graph-store backend abstraction for huge graphs, embedding batching, durable extraction queue (retries).
- **Where:** `infrastructure/db/`, `pipeline/async_extractor.py`.
- **Acceptance:** load test 1k users; extraction queue survives restart.

### E26. Backup, PITR & data sovereignty (self-hostable)
- **Inspired by:** user preference for self-hostable / data-sovereign tools.
- **Current state:** raw Postgres; no documented backup/PITR.
- **What to build:** WAL archival + PITR runbook, encrypted snapshots, portable JSON export/import of a user's full memory graph.
- **Where:** `infrastructure/`, new `scripts/export_memory.py`.
- **Acceptance:** full user memory exported → re-imported identically.

---

## Recommended execution order

| Rank | Item | Why |
|------|------|-----|
| 1 | **E22 AuthN/tenant isolation** | blocks real deployment; security-critical |
| 2 | **E4 Chunking + structured extraction** | raises quality across all layers |
| 3 | **E7 Scheduled sleep consolidation** | turns dream cycle into a real brain-like sleep stage |
| 4 | **E10 Bitemporal contradiction lattice** | truthfulness + temporal recall (Zep's 71% edge) |
| 5 | **E11/E13/E14 Belief + forgetting + salience** | without these memory grows unbounded and loses signal |
| 6 | **E16 Metacognitive routing** | intelligent per-intent retrieval |
| 7 | **E21 Eval harness** | can't claim "production" without measured recall/faithfulness |
| 8 | **E23 Rate-limit + cost guardrails** | needed before multi-user load |
| 9 | **E8 A-MEM evolution** | living, self-updating memory graph |
| 10 | **E19 Cross-session continuity** | lifetime memory property |
| 11 | **E2/E3/E5/E6/E9/E12/E15/E17/E18/E20** | depth: compression, procedural, recognition, multimodal, spacing, provenance, reconsolidation, self-improve, abstention, temporal |
| 12 | **E24/E25/E26** | scale, observe, recover |

---

## "Done" criteria — a memory system that works like a human brain

- [ ] **Encode:** text + docs + multimodal, chunked, cross-chunk resolution, cheap recognition pre-filter, **prospective + affective + sensory buffers** present.
- [ ] **Store:** episodic / semantic / procedural / in-context **+ prospective / affective / implicit / meta / sensory** as first-class types.
- [ ] **Admit:** scored write-gate (novelty×importance×credibility) prevents bloat at source.
- [ ] **Consolidate:** automatic sleep-time replay (reflect → contradict → strengthen → compress → prune) + A-MEM evolution + **rehearsal buffer** (anti-catastrophic-forgetting).
- [ ] **Believe:** bitemporal contradiction lattice + Bayesian belief; retrieval surfaces provenance + calibrated uncertainty + **source trust**.
- [ ] **Forget:** algorithmic forgetting + spaced reinforcement; protected core resists decay/overwrite (stability–plasticity).
- [ ] **Retrieve:** metacognitively routed per intent; recognition + PPR; temporal/causal queries; **working-memory eviction** under capacity.
- [ ] **Persist:** cross-session user continuity + lifetime narrative; **prospective intentions fire on cue**.
- [ ] **Secure:** authenticated, tenant-isolated, poisoning-quarantined, **unlearn/right-to-be-forgotten**, tool-call memory sandboxed.
- [ ] **Measure:** Recall@K, faithfulness, temporal accuracy, prospective-fire accuracy on golden sets.
- [ ] **Deploy:** rate-limited, observable, backed-up, horizontally scalable, survives provider outages.

*37 enhancements (E1–E37). Single source of truth for AGI-memory enhancements. Alignment gaps are in `ARCHITECTURE_REFERENCE.md` (all RESOLVED).*
