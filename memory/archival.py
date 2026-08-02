"""Layer 6 (Letta): Archival memory — cold store for off-context memories recalled on demand.

Letta keeps core (hot, in-prompt) memory separate from archival (cold) memory that is
searched and injected only when relevant. This module implements the archival store +
recall, plus a LightRAG-style dual-level query router (naive / local / global / hybrid).
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from infrastructure.db import DatabasePool
from infrastructure.llm import FastEmbeddingProvider
from memory.atomic import search_facts, search_facts_dual_level
from config import settings
from utils import measure_latency, log_atomic, log_error
from core.exceptions import StorageOperationError

embedder = FastEmbeddingProvider()


async def archive_memory(user_id: str, content: str, agent_id: Optional[str] = None) -> UUID:
    """Store a piece of cold memory (archival). Returns the new id."""
    async with measure_latency("memory.archival.archive_memory"):
        try:
            embedding = await embedder.embed_text(content)
            vec = f"[{','.join(str(x) for x in embedding)}]"
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO archival_memory (user_id, agent_id, content, embedding)
                    VALUES ($1, $2, $3, $4::vector)
                    RETURNING id;
                    """,
                    user_id, agent_id, content, vec,
                )
                assert row is not None
                return row["id"]
        except Exception as e:
            log_error(f"Archive failed: {e}")
            raise StorageOperationError(f"Archive memory error: {e}")


async def recall_archival(user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Recall the most relevant archival memories for a query (vector search)."""
    async with measure_latency("memory.archival.recall_archival"):
        try:
            embedding = await embedder.embed_text(query)
            vec = f"[{','.join(str(x) for x in embedding)}]"
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, content, 1 - (embedding <=> $2::vector) AS sim
                    FROM archival_memory
                    WHERE user_id = $1 AND embedding IS NOT NULL
                    ORDER BY embedding <=> $2::vector ASC
                    LIMIT $3;
                    """,
                    user_id, vec, limit,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            log_error(f"Recall failed: {e}")
            return []


# LightRAG query modes: naive (vector only), local (entity/core facts),
# global (graph/community summary), hybrid (both).
QUERY_MODES = ("naive", "local", "global", "hybrid")


async def retrieve_memory(
    user_id: str,
    query: str,
    mode: str = "hybrid",
    limit: int = settings.TOP_K_FACTS,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """LightRAG dual-level retrieval router.

    - naive:  pure vector search over atomic facts (no graph).
    - local:  entity-centric — atomic facts + low-level graph relationships.
    - global: relationship/community-centric — graph summary + reflections.
    - hybrid: local + global combined (default).
    """
    if mode not in QUERY_MODES:
        mode = "hybrid"

    if mode == "naive":
        facts = await search_facts(user_id, query, limit=limit, session_id=session_id, agent_id=agent_id)
        return {"mode": "naive", "facts": [f.model_dump() for f in facts], "high_level": {}}

    dual = await search_facts_dual_level(user_id, query, limit=limit, session_id=session_id, agent_id=agent_id)
    low = dual["low_level_facts"]
    high = dual["high_level_concepts"]

    if mode == "local":
        return {"mode": "local", "facts": low, "high_level": {}}
    if mode == "global":
        return {"mode": "global", "facts": [], "high_level": high}
    # hybrid
    return {"mode": "hybrid", "facts": low, "high_level": high}
