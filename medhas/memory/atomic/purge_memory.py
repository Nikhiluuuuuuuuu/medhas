"""Mem0 & Cognee Purge API: Clean database state wipe for test isolation."""

from medhas.storage import DatabasePool
from medhas.utils import measure_latency, logger

async def purge_user_memories(user_id: str) -> None:
    """Wipe all database records for a given user_id to ensure clean test isolation."""
    async with measure_latency("memory.atomic.purge_user_memories"):
        try:
            async with DatabasePool.acquire() as conn:
                await conn.execute("DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE user_id = $1);", user_id)
                await conn.execute("DELETE FROM sessions WHERE user_id = $1;", user_id)
                await conn.execute("DELETE FROM working_memory WHERE user_id = $1;", user_id)
                await conn.execute("DELETE FROM atomic_facts WHERE user_id = $1;", user_id)
                await conn.execute("DELETE FROM graph_edges WHERE user_id = $1;", user_id)
                await conn.execute("DELETE FROM graph_nodes WHERE user_id = $1;", user_id)
                logger.info(f"🧹 Cleaned database state for user: {user_id}")
        except Exception as e:
            logger.error(f"Error purging user memories for {user_id}: {e}")
