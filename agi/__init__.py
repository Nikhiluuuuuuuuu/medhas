"""Medhas AGI memory layer (roadmap 0–37).

Public surface: `from agi import engine, MemoryEngine` plus the per-capability modules
under agi.* . All behaviour is ADDITIVE — existing memory/ and pipeline/ modules are
never modified except for one import line; everything new lives here.
"""

from agi.engine import MemoryEngine, engine

from agi.memory_types import MemoryType, route, is_valid
from agi.admission import evaluate_admission
from agi.bitemporal import (
    bayesian_update, revise_fact_belief, invalidate_fact, mark_contradiction,
    facts_valid_at, fact_provenance,
)
from agi.forgetting import (
    retention, run_forgetting_sweep, reconsolidate, set_affect,
    protect_core_memories, is_protected, schedule_review, due_for_review,
)
from agi.security import (
    sign_write, verify_write, trust_for_source, check_poisoning, quarantine_fact,
    release_quarantine, list_quarantined, forget, sandbox_for_tools,
)
from agi.consolidation import (
    compress_episodes, induce_skills, evolve_memory_network, run_consolidation,
)
from agi.scheduler import scheduler, rehearse
from agi.interference import (
    interference_matrix, resolve_interference, evict_working_memory, recognize, eviction_scores,
)
from agi.metacognitive import (
    route_query, should_abstain, RetrievalOutcome, critic, log_memory_event,
)
from agi.usermodel import (
    build_user_model, get_user_model, timeline, what_changed, why_chain,
)
from agi.prospective import (
    add_intention, check_cues, complete_intention, list_intentions,
)
from agi.metamemory import assess as metamemory_assess, known_unknowns, knowledge_map
from agi.sensory import (
    buffer_percept, promote_percepts, sweep_expired, list_buffer, attention_filter,
)
from agi.ingest import (
    chunk_text, extract_facts, extract_from_document, detect_duplicates,
)
from agi.auth import authenticate, authorize, rate_limiter, generate_api_key
from agi.scaling import ensure_hot_view, refresh_hot_view, partition_report
from agi.export import export_user_memory, export_to_file, import_user_memory
from agi.eval import run_eval_suite, temporal_consistency_check, EvalCase

__all__ = [
    "MemoryEngine", "engine", "MemoryType", "route", "is_valid",
    "evaluate_admission", "bayesian_update", "revise_fact_belief", "invalidate_fact",
    "mark_contradiction", "facts_valid_at", "fact_provenance", "retention",
    "run_forgetting_sweep", "reconsolidate", "set_affect", "protect_core_memories",
    "is_protected", "schedule_review", "due_for_review", "sign_write", "verify_write",
    "trust_for_source", "check_poisoning", "quarantine_fact", "release_quarantine",
    "list_quarantined", "forget", "sandbox_for_tools", "compress_episodes",
    "induce_skills", "evolve_memory_network", "run_consolidation", "scheduler",
    "rehearse", "interference_matrix", "resolve_interference", "evict_working_memory",
    "recognize", "eviction_scores", "route_query", "should_abstain", "RetrievalOutcome",
    "critic", "log_memory_event", "build_user_model", "get_user_model", "timeline",
    "what_changed", "why_chain", "add_intention", "check_cues", "complete_intention",
    "list_intentions", "metamemory_assess", "known_unknowns", "knowledge_map",
    "buffer_percept", "promote_percepts", "sweep_expired", "list_buffer",
    "attention_filter", "chunk_text", "extract_facts", "extract_from_document",
    "detect_duplicates", "authenticate", "authorize", "rate_limiter",
    "generate_api_key", "ensure_hot_view", "refresh_hot_view", "partition_report",
    "export_user_memory", "export_to_file", "import_user_memory", "run_eval_suite",
    "temporal_consistency_check", "EvalCase",
]
