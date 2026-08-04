"""Database Schema Initialization & Migration Module."""

import os
from medhas.storage.connection import DatabasePool
from medhas.utils import logger, measure_latency
from medhas.core.exceptions import StorageOperationError

INIT_SQL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "schema.sql")

async def initialize_schema() -> None:
    """Read schema.sql and execute table setup DDL via asyncpg.

    Embedding columns are resized to the TRUE dimension of the configured embedding
    model (not a hard-coded 768). This prevents the 'expected 768 dimensions, not 384'
    failure when a different-dimension model is configured, and migrates existing
    columns in place.
    """
    if not os.path.exists(INIT_SQL_PATH):
        raise StorageOperationError(f"schema.sql file not found at {INIT_SQL_PATH}")

    with open(INIT_SQL_PATH, "r", encoding="utf-8") as f:
        sql_script = f.read()

    # Detect the real embedding dimension from the model before touching the schema.
    from medhas.embeddings.embedding_provider import get_embedding_dimension
    dim = get_embedding_dimension()

    try:
        async with measure_latency("initialize_schema"):
            async with DatabasePool.acquire() as conn:
                await conn.execute(sql_script)
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS importance_score FLOAT NOT NULL DEFAULT 5.0;")
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS session_id UUID NULL;")
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS agent_id VARCHAR(255) NULL;")
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64) NULL;")
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS run_id VARCHAR(255) NULL;")
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS categories TEXT[] NULL;")
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS memory_type VARCHAR(50) NOT NULL DEFAULT 'semantic';")
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_atomic_facts_user_hash ON atomic_facts(user_id, content_hash) WHERE content_hash IS NOT NULL;")
                await conn.execute("ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS session_id UUID NULL;")
                await conn.execute("ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS agent_id VARCHAR(255) NULL;")
                await conn.execute(f"ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS embedding vector({dim}) NULL;")
                await conn.execute("ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS fact_ids UUID[] NULL;")
                await conn.execute("ALTER TABLE graph_edges ADD COLUMN IF NOT EXISTS session_id UUID NULL;")
                await conn.execute("ALTER TABLE graph_edges ADD COLUMN IF NOT EXISTS agent_id VARCHAR(255) NULL;")
                await conn.execute("ALTER TABLE graph_edges ADD COLUMN IF NOT EXISTS link_type VARCHAR(100) NULL;")
                await conn.execute("ALTER TABLE graph_edges ADD COLUMN IF NOT EXISTS link_source VARCHAR(50) NOT NULL DEFAULT 'extracted';")
                await conn.execute("CREATE TABLE IF NOT EXISTS episodes (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id VARCHAR(255) NOT NULL, session_id UUID NULL, agent_id VARCHAR(255) NULL, content TEXT NOT NULL, source VARCHAR(50) NOT NULL DEFAULT 'message', reference_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);")
                # Graphiti episode resolution layer: source_description + group_id (graphiti_core/graphiti.py:980 add_episode)
                await conn.execute("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS source_description VARCHAR(255) NULL;")
                await conn.execute("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS group_id VARCHAR(255) NULL;")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user_time ON episodes(user_id, reference_time DESC);")
                await conn.execute(f"CREATE TABLE IF NOT EXISTS archival_memory (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id VARCHAR(255) NOT NULL, agent_id VARCHAR(255) NULL, content TEXT NOT NULL, embedding vector({dim}) NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_atomic_facts_fts ON atomic_facts USING gin(to_tsvector('english', fact_text));")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_archival_user ON archival_memory(user_id);")
                # Letta recall tier: semantic recall over conversation history (memory/session/recall.py)
                await conn.execute(f"ALTER TABLE messages ADD COLUMN IF NOT EXISTS embedding vector({dim}) NULL;")
                # Resize any existing embedding columns to the detected dimension (idempotent migration).
                for table in ("graph_nodes", "archival_memory", "messages"):
                    await conn.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector({dim});")
                await conn.execute("DROP INDEX IF EXISTS idx_messages_embedding;")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_messages_embedding ON messages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
        # AGI memory roadmap (E1–E37): additive schema extensions, idempotent.
        from medhas.storage.agi_schema import initialize_agi_schema
        await initialize_agi_schema(dim=dim)
        logger.info(f"✅ Database schema & vector HNSW indexes initialized (embedding dim={dim}).")
    except Exception as e:
        logger.error(f"❌ Failed to execute schema DDL: {e}")
        raise StorageOperationError(f"Schema DDL execution failed: {e}")
