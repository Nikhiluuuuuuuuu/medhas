-- Enable pgvector extension for semantic similarity search
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. BASE LAYER: Sessions and Chat Transcripts (Convex-style immutable logs)
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, created_at ASC);

-- 2. WORKING MEMORY LAYER: Dynamic Prompt Block RAM (Letta-style JSONB RAM)
CREATE TABLE IF NOT EXISTS working_memory (
    user_id VARCHAR(255) PRIMARY KEY,
    blocks JSONB NOT NULL DEFAULT '{
        "persona": "I am an advanced AI assistant powered by a 4-in-1 Unified Local Memory Engine.",
        "user_profile": "User profile is currently empty.",
        "scratchpad": "No active scratchpad notes.",
        "active_goals": "No active goals set."
    }'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 3. ATOMIC FACT LAYER: Preferences & Deduplication (Mem0 & GBrain vector lifecycle)
CREATE TABLE IF NOT EXISTS atomic_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    session_id UUID NULL,
    agent_id VARCHAR(255) NULL,
    fact_text TEXT NOT NULL,
    embedding vector(384),
    importance_score FLOAT NOT NULL DEFAULT 5.0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expired_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_atomic_facts_user_active ON atomic_facts(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_atomic_facts_fts ON atomic_facts USING gin(to_tsvector('english', fact_text));
CREATE INDEX IF NOT EXISTS atomic_facts_vector_idx 
ON atomic_facts USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. TEMPORAL GRAPH LAYER: Entity Relationships (Zep / Graphiti bi-temporal graph)
CREATE TABLE IF NOT EXISTS graph_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    session_id UUID NULL,
    agent_id VARCHAR(255) NULL,
    name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(100) NOT NULL, -- 'Person', 'Company', 'Location', 'Project', etc.
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(384) NULL,  -- node embedding for semantic entity merge (Graphiti-style)
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    session_id UUID NULL,
    agent_id VARCHAR(255) NULL,
    source_id UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    relationship VARCHAR(255) NOT NULL,
    link_type VARCHAR(100) NULL,   -- GBrain-style typed link (e.g. 'relates_to', 'works_at', 'founded')
    link_source VARCHAR(50) NOT NULL DEFAULT 'extracted',  -- provenance: 'manual' | 'extracted' | 'inferred'
    valid_from TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    valid_to TIMESTAMPTZ NULL, -- NULL indicates currently active/valid edge
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expired_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_user_valid 
ON graph_edges(user_id, valid_from, valid_to);

-- 5b. EPISODES LAYER (Cognee/Graphiti): anchors for background extraction runs.
-- Each conversation turn (or batched turns) becomes an episode that facts/edges derive from.
CREATE TABLE IF NOT EXISTS episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    session_id UUID NULL,
    agent_id VARCHAR(255) NULL,
    content TEXT NOT NULL,
    source VARCHAR(50) NOT NULL DEFAULT 'message',  -- 'message' | 'document'
    reference_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_episodes_user_time ON episodes(user_id, reference_time DESC);

-- 6. ARCHIVAL MEMORY LAYER (Letta): cold store for off-context memories recalled on demand.
CREATE TABLE IF NOT EXISTS archival_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255) NULL,
    content TEXT NOT NULL,
    embedding vector(384) NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_archival_user ON archival_memory(user_id);
CREATE INDEX IF NOT EXISTS archival_memory_vector_idx
ON archival_memory USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
