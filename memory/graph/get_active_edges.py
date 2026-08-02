"""Layer 4 (Zep/Graphiti): Return currently-active (valid_to IS NULL) edges for a node."""

from typing import List, Dict, Any, Optional
from uuid import UUID
from infrastructure.db import DatabasePool
from utils import measure_latency, log_error


async def get_active_edges(user_id: str, node_id: UUID) -> List[Dict[str, Any]]:
    """Return active outgoing/incoming edges for a node (bi-temporal: valid_to IS NULL)."""
    async with measure_latency("memory.graph.get_active_edges"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT e.id, e.source_id, e.target_id, e.relationship, e.valid_from, e.valid_to
                    FROM graph_edges e
                    WHERE e.user_id = $1 AND e.valid_to IS NULL
                      AND (e.source_id = $2 OR e.target_id = $2);
                    """,
                    user_id, node_id,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            log_error(f"get_active_edges failed: {e}")
            return []
