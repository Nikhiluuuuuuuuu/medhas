"""Database Schema Initialization & Migration Module."""

import os
from infrastructure.db.connection import DatabasePool
from utils import logger, measure_latency
from core.exceptions import StorageOperationError

INIT_SQL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "schema.sql")

async def initialize_schema() -> None:
    """Read schema.sql and execute table setup DDL via asyncpg."""
    if not os.path.exists(INIT_SQL_PATH):
        raise StorageOperationError(f"schema.sql file not found at {INIT_SQL_PATH}")

    with open(INIT_SQL_PATH, "r", encoding="utf-8") as f:
        sql_script = f.read()

    try:
        async with measure_latency("initialize_schema"):
            async with DatabasePool.acquire() as conn:
                await conn.execute(sql_script)
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS importance_score FLOAT NOT NULL DEFAULT 5.0;")
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS session_id UUID NULL;")
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS agent_id VARCHAR(255) NULL;")
                await conn.execute("ALTER TABLE atomic_facts ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64) NULL;")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_atomic_facts_user_hash ON atomic_facts(user_id, content_hash) WHERE content_hash IS NOT NULL;")
                await conn.execute("ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS session_id UUID NULL;")
                await conn.execute("ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS agent_id VARCHAR(255) NULL;")
                await conn.execute("ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS embedding vector(384) NULL;")
                await conn.execute("ALTER TABLE graph_edges ADD COLUMN IF NOT EXISTS session_id UUID NULL;")
                await conn.execute("ALTER TABLE graph_edges ADD COLUMN IF NOT EXISTS agent_id VARCHAR(255) NULL;")
                await conn.execute("CREATE TABLE IF NOT EXISTS episodes (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id VARCHAR(255) NOT NULL, session_id UUID NULL, agent_id VARCHAR(255) NULL, content TEXT NOT NULL, source VARCHAR(50) NOT NULL DEFAULT 'message', reference_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);")
                await conn.execute("CREATE TABLE IF NOT EXISTS archival_memory (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id VARCHAR(255) NOT NULL, agent_id VARCHAR(255) NULL, content TEXT NOT NULL, embedding vector(384) NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_atomic_facts_fts ON atomic_facts USING gin(to_tsvector('english', fact_text));")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_archival_user ON archival_memory(user_id);")
        logger.info("✅ Database schema & vector HNSW indexes initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to execute schema DDL: {e}")
        raise StorageOperationError(f"Schema DDL execution failed: {e}")
