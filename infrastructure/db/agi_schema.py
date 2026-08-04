"""AGI memory schema extensions (roadmap E1–E37).

ADDITIVE ONLY. This module never drops or rewrites existing objects; it adds the
columns and tables the AGI-memory roadmap requires, using IF NOT EXISTS so it is
idempotent and safe to run on an existing database.

Covered:
  E1  memory_type routing            -> atomic_facts.memory_type (exists) + episodes/percepts
  E9  spaced reinforcement           -> next_review_at, review_interval_days, review_count
  E10 bitemporal lattice             -> valid_from / valid_to / invalidated_by on facts
  E11 belief revision                -> belief_confidence
  E12 provenance                     -> source_episode_id, contradicted_by
  E13 forgetting                     -> decay_half_life_days, last_accessed_at
  E14 salience                       -> access_count, salience_score
  E28 affective memory               -> affect_valence, affect_arousal
  E30 implicit memory                -> provenance_kind (explicit|implicit_inferred)
  E32 admission control              -> admission_score
  E33 interference / protected core  -> is_protected
  E34 security                       -> source_trust, is_quarantined, write_signature
  E2  episodic->semantic compression -> episodes.compressed, episodes.gist_fact_id
  E27 prospective memory             -> prospective_memory table
  E29 meta-memory                    -> meta_memory table
  E31 sensory buffer                 -> percept_buffer table
  E21 eval harness                   -> eval_runs table
  E24 observability                  -> memory_events table
"""

from infrastructure.db.connection import DatabasePool
from config import settings
from utils import logger

FACT_COLUMNS = [
    # (column, DDL type/default)
    ("valid_from", "TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"),
    ("valid_to", "TIMESTAMPTZ NULL"),
    ("invalidated_by", "UUID NULL"),
    ("belief_confidence", "FLOAT NOT NULL DEFAULT 0.70"),
    ("source_episode_id", "UUID NULL"),
    ("contradicted_by", "UUID[] NULL"),
    ("last_accessed_at", "TIMESTAMPTZ NULL"),
    ("access_count", "INTEGER NOT NULL DEFAULT 0"),
    ("salience_score", "FLOAT NOT NULL DEFAULT 0.50"),
    ("decay_half_life_days", "FLOAT NOT NULL DEFAULT 7.0"),
    ("next_review_at", "TIMESTAMPTZ NULL"),
    ("review_interval_days", "FLOAT NOT NULL DEFAULT 1.0"),
    ("review_count", "INTEGER NOT NULL DEFAULT 0"),
    ("affect_valence", "FLOAT NOT NULL DEFAULT 0.0"),
    ("affect_arousal", "FLOAT NOT NULL DEFAULT 0.0"),
    ("provenance_kind", "VARCHAR(32) NOT NULL DEFAULT 'explicit'"),
    ("admission_score", "FLOAT NOT NULL DEFAULT 1.0"),
    ("is_protected", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ("source_trust", "FLOAT NOT NULL DEFAULT 0.80"),
    ("is_quarantined", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ("write_signature", "VARCHAR(64) NULL"),
    ("note", "TEXT NULL"),
    ("linked_ids", "UUID[] NULL"),
]

TABLES = [
    # E27 prospective memory
    """
    CREATE TABLE IF NOT EXISTS prospective_memory (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id VARCHAR(255) NOT NULL,
        agent_id VARCHAR(255) NULL,
        intent TEXT NOT NULL,
        cue_text TEXT NULL,
        cue_kind VARCHAR(20) NOT NULL DEFAULT 'event',   -- event | time
        trigger_at TIMESTAMPTZ NULL,
        fired_at TIMESTAMPTZ NULL,
        is_done BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_prospective_user ON prospective_memory(user_id, is_done);",
    # E29 meta-memory
    """
    CREATE TABLE IF NOT EXISTS meta_memory (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id VARCHAR(255) NOT NULL,
        topic VARCHAR(255) NOT NULL,
        known_count INTEGER NOT NULL DEFAULT 0,
        mean_confidence FLOAT NOT NULL DEFAULT 0.0,
        is_known_unknown BOOLEAN NOT NULL DEFAULT FALSE,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, topic)
    );
    """,
    # E31 sensory / perceptual buffer
    """
    CREATE TABLE IF NOT EXISTS percept_buffer (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id VARCHAR(255) NOT NULL,
        session_id UUID NULL,
        modality VARCHAR(32) NOT NULL DEFAULT 'text',
        raw_caption TEXT NOT NULL,
        embedding vector(768) NULL,
        expires_at TIMESTAMPTZ NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_percept_user_exp ON percept_buffer(user_id, expires_at DESC);",
    # E19 user profile / lifetime narrative
    """
    CREATE TABLE IF NOT EXISTS user_profile (
        user_id VARCHAR(255) PRIMARY KEY,
        profile TEXT NOT NULL DEFAULT '',
        narrative TEXT NOT NULL DEFAULT '',
        traits JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    # E21 eval harness runs
    """
    CREATE TABLE IF NOT EXISTS eval_runs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id VARCHAR(255) NOT NULL,
        suite VARCHAR(100) NOT NULL,
        metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    # E24 observability event log
    """
    CREATE TABLE IF NOT EXISTS memory_events (
        id BIGSERIAL PRIMARY KEY,
        user_id VARCHAR(255) NULL,
        event VARCHAR(80) NOT NULL,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_events_time ON memory_events(created_at DESC);",
    # E36 rehearsal buffer
    """
    CREATE TABLE IF NOT EXISTS rehearsal_buffer (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id VARCHAR(255) NOT NULL,
        fact_id UUID NOT NULL,
        last_rehearsed_at TIMESTAMPTZ NULL,
        rehearsal_count INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, fact_id)
    );
    """,
    # E22 API keys / tenants
    """CREATE TABLE IF NOT EXISTS api_keys (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        key_hash VARCHAR(64) NOT NULL UNIQUE,
        user_id VARCHAR(255) NOT NULL,
        tenant_id VARCHAR(255) NOT NULL DEFAULT 'default',
        scopes TEXT[] NOT NULL DEFAULT ARRAY['read','write'],
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    # E-cognition: embodiment / body action-effect model
    """
    CREATE TABLE IF NOT EXISTS body_effects (
        user_id    TEXT NOT NULL,
        action     TEXT NOT NULL,
        context    TEXT NOT NULL,
        outcome    TEXT NOT NULL,
        success    BOOLEAN NOT NULL DEFAULT TRUE,
        count      INTEGER NOT NULL DEFAULT 1,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (user_id, action, context, outcome)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_body_effects_user_action ON body_effects(user_id, action);",
]

EPISODE_COLUMNS = [
    ("compressed", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ("gist_fact_id", "UUID NULL"),
    ("memory_type", "VARCHAR(50) NOT NULL DEFAULT 'episodic'"),
    ("affect_valence", "FLOAT NOT NULL DEFAULT 0.0"),
    ("affect_arousal", "FLOAT NOT NULL DEFAULT 0.0"),
]

EDGE_COLUMNS = [
    ("belief_confidence", "FLOAT NOT NULL DEFAULT 0.70"),
    ("invalidated_by", "UUID NULL"),
    ("source_episode_id", "UUID NULL"),
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_atomic_facts_review ON atomic_facts(user_id, next_review_at) WHERE is_active = TRUE;",
    "CREATE INDEX IF NOT EXISTS idx_atomic_facts_valid ON atomic_facts(user_id, valid_from, valid_to);",
    "CREATE INDEX IF NOT EXISTS idx_atomic_facts_type ON atomic_facts(user_id, memory_type) WHERE is_active = TRUE;",
]


async def initialize_agi_schema() -> None:
    """Apply all additive AGI schema extensions. Idempotent."""
    async with DatabasePool.acquire() as conn:
        for col, ddl in FACT_COLUMNS:
            await conn.execute(f"ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS {col} {ddl};")
        for col, ddl in EPISODE_COLUMNS:
            await conn.execute(f"ALTER TABLE episodes ADD COLUMN IF NOT EXISTS {col} {ddl};")
        for col, ddl in EDGE_COLUMNS:
            await conn.execute(f"ALTER TABLE graph_edges ADD COLUMN IF NOT EXISTS {col} {ddl};")
        for stmt in TABLES:
            await conn.execute(stmt)
        for stmt in INDEXES:
            await conn.execute(stmt)
    if settings.FACT_RERANKER_WARMUP and settings.FACT_RERANKER_ENABLED:
        try:
            from memory.atomic.reranker import warmup_reranker
            warmup_reranker()
        except Exception:
            # best-effort; the reranker lazy-loads on first query regardless
            pass
    logger.info("✅ AGI memory schema extensions (E1–E37) applied.")
