"""Layer 4 (Zep/Graphiti): Point-in-time temporal graph relationship query."""

from datetime import datetime
from typing import List, Dict, Any
from infrastructure.db import DatabasePool
from utils import measure_latency, logger
from core.exceptions import StorageOperationError

async def query_point_in_time(
    user_id: str,
    entity_name: str,
    target_time: datetime
) -> List[Dict[str, Any]]:
    """Execute point-in-time temporal query for entity relationships valid at target_time."""
    async with measure_latency(f"memory.graph.query_point_in_time ({entity_name} @ {target_time})"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT 
                        sn.name AS source_name,
                        tn.name AS target_name,
                        e.relationship,
                        e.valid_from,
                        e.valid_to
                    FROM graph_edges e
                    JOIN graph_nodes sn ON e.source_id = sn.id
                    JOIN graph_nodes tn ON e.target_id = tn.id
                    WHERE e.user_id = $1 
                      AND (LOWER(sn.name) = LOWER($2) OR LOWER(tn.name) = LOWER($2))
                      AND e.valid_from <= $3
                      AND (e.valid_to IS NULL OR e.valid_to > $3);
                    """,
                    user_id,
                    entity_name,
                    target_time
                )
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed point-in-time graph query: {e}")
            raise StorageOperationError(f"Point-in-time query error: {e}")
