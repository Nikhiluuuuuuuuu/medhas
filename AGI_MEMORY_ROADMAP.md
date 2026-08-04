# Medhas — "Superpower" AGI Memory System (Roadmap 0–37)

This implements every item in `docs/roadmap.md` as an **additive** memory layer on top
of the existing 6-in-1 engine (MedQA proven, Zep/Graphiti temporal, Mem0 atomic, Letta
working/archival, Generative, A-MEM). All new behaviour lives under `agi/` — the
original `memory/`, `pipeline/`, and `server.py` routes are **not rewritten**, only
extended by one include line.

## How to run
```bash
source /home/ubuntu/.venv-medhas/bin/activate
POSTGRES_DB=medhas_test python -m pytest tests/ -q   # 39 tests, all green
uvicorn server:app --port 8000                       # new /agi/* endpoints mounted
```

## What was implemented (mapping to roadmap)

| # | Capability | Module / Hook | Key API |
|---|-----------|---------------|---------|
| 0 | CoALA taxonomy + store routing | `agi/memory_types.py` | `route("semantic") -> atomic_facts` |
| 1 | First-class memory types | `agi/memory_types.py` | 9 types incl. prospective/affective/implicit/meta/sensory |
| 2 | Episodic→semantic compression | `agi/consolidation.py` | `compress_episodes()` |
| 3 | Procedural auto-induction | `agi/consolidation.py` | `induce_skills()` |
| 4 | Chunking + fact extraction | `agi/ingest.py` | `chunk_text()`, `extract_from_document()` |
| 5 | Recognition-before-recall | `agi/interference.py` | `recognize()` (content-hash gate) |
| 6 | Multimodal ingestion | `agi/sensory.py` | `buffer_percept()` (text/image/audio/doc/tool) |
| 7 | Consolidation scheduler | `agi/scheduler.py` + `agi/consolidation.py` | `scheduler.run_once()`, `run_consolidation()` |
| 8 | A-MEM memory evolution | `agi/consolidation.py` | `evolve_memory_network()` (retro-link + revise) |
| 9 | Spaced reinforcement | `agi/forgetting.py` | `schedule_review()`, `due_for_review()` |
| 10 | Bitemporal contradiction lattice | `agi/bitemporal.py` | `facts_valid_at()`, `invalidate_fact()` |
| 11 | Belief revision (Bayesian) | `agi/bitemporal.py` | `revise_fact_belief()`, `bayesian_update()` |
| 12 | Provenance + uncertainty | `agi/bitemporal.py` | `fact_provenance()` |
| 13 | Algorithmic forgetting | `agi/forgetting.py` | `run_forgetting_sweep()` (Ebbinghaus) |
| 14 | Salience learning | `agi/forgetting.py` | `reconsolidate()` (testing-effect) |
| 15 | Reconsolidation | `agi/forgetting.py` | `reconsolidate()` |
| 16 | Metacognitive retrieval routing | `agi/metacognitive.py` | `route_query()` (cheapest-sufficient) |
| 17 | Self-improvement critic | `agi/metacognitive.py` | `critic.score()`, `SelfImprovementCritic` |
| 18 | Calibrated abstention | `agi/metacognitive.py` + `agi/metamemory.py` | `should_abstain()` |
| 19 | Lifetime user model + narrative | `agi/usermodel.py` | `build_user_model()`, `get_user_model()` |
| 20 | Temporal + causal API | `agi/usermodel.py` | `timeline()`, `what_changed()`, `why_chain()` |
| 21 | Eval harness | `agi/eval.py` | `run_eval_suite()` (LOCOMO/SITUATEDQA-style) |
| 22 | Multi-tenant API auth | `agi/auth.py` | `authenticate()`, `generate_api_key()` |
| 23 | Rate limiting | `agi/auth.py` | `rate_limiter.allow()` (token bucket) |
| 24 | Observability | `agi/metacognitive.py` | `log_memory_event()` → `memory_events` |
| 25 | Scalability / hot views | `agi/scaling.py` | `ensure_hot_view()`, `partition_report()` |
| 26 | Backup / export | `agi/export.py` | `export_user_memory()`, `import_user_memory()` |
| 27 | Prospective memory | `agi/prospective.py` | `add_intention()`, `check_cues()` |
| 28 | Affective memory | `agi/forgetting.py` | `set_affect()` (arousal → flat decay) |
| 29 | Meta-memory (known-unknowns) | `agi/metamemory.py` | `assess()`, `known_unknowns()` |
| 30 | Implicit memory | `agi/admission.py` | `provenance_kind='implicit_inferred'` |
| 31 | Sensory / percept buffer | `agi/sensory.py` | `buffer_percept()`, `promote_percepts()` |
| 32 | Adaptive admission control | `agi/admission.py` | `evaluate_admission()` (novelty+importance+credibility) |
| 33 | Protected core (interference) | `agi/forgetting.py` | `protect_core_memories()`, `is_protected()` |
| 34 | Security (poisoning/unlearning) | `agi/security.py` | `check_poisoning()`, `forget()`, `sandbox_for_tools()` |
| 35 | Working-memory eviction | `agi/interference.py` | `evict_working_memory()` (to archival) |
| 36 | Rehearsal buffer | `agi/scheduler.py` | `rehearse()` (experience replay) |
| 37 | Interference matrix | `agi/interference.py` | `interference_matrix()`, `resolve_interference()` |

## Unified facade
`agi/engine.py` exposes `MemoryEngine` with two entry points used by the API:
- `engine.remember(...)` — admission → type routing → contradiction/Bayesian belief →
  A-MEM evolution → security/quarantine, all in one call.
- `engine.recall(...)` — recognition gate → routing → reconsolidation → tool sandbox →
  calibrated abstention.

## Verification
- `tests/test_agi_memory_roadmap.py` — 15 tests covering admission, bayesian belief,
  bitemporal contradiction, forgetting/protected core, prospective firing, sensory
  buffer, export round-trip, eval temporal-consistency, spaced review, affect.
- Full suite: **39 passed** (24 original + 15 new) against real PostgreSQL (`medhas_test`).
- Live end-to-end run confirmed recall, prospective firing, user-model build, and export.

## Operational notes
- LLM-backed steps (compression, skill induction, user-model, A-MEM evolution, why-chain)
  require the configured Groq key; they degrade gracefully (deterministic fallbacks) when
  rate-limited, so the pipeline never blocks on the model.
- New DB objects are additive (`infrastructure/db/agi_schema.py`, idempotent
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` + new tables) — safe to re-run.
