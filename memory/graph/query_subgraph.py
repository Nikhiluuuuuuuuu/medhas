"""Layer 4 (Zep/Graphiti): Retrieve active entity subgraph for entity names."""

import json
from typing import List, Dict, Any, Optional
from infrastructure.db import DatabasePool
from schemas import GraphNodeSchema, SubgraphQueryResult
from utils import measure_latency, logger
from core.exceptions import StorageOperationError

async def query_subgraph(user_id: str, entity_name: str) -> Optional[SubgraphQueryResult]:
    """Fetch node and currently active valid edges for an entity name."""
    async with measure_latency(f"memory.graph.query_subgraph ({entity_name})"):
        try:
            async with DatabasePool.acquire() as conn:
                # 1. Fetch node
                node_row = await conn.fetchrow(
                    """
                    SELECT id, user_id, name, entity_type, attributes, created_at
                    FROM graph_nodes
                    WHERE user_id = $1 AND LOWER(name) = LOWER($2);
                    """,
                    user_id,
                    entity_name
                )
                if not node_row:
                    return None

                node = GraphNodeSchema(
                    id=node_row["id"],
                    user_id=node_row["user_id"],
                    name=node_row["name"],
                    entity_type=node_row["entity_type"],
                    attributes=json.loads(node_row["attributes"]) if isinstance(node_row["attributes"], str) else dict(node_row["attributes"]),
                    created_at=node_row["created_at"]
                )

                # 2. Fetch active outgoing edges (valid_to IS NULL)
                out_rows = await conn.fetch(
                    """
                    SELECT e.relationship, e.valid_from, n.name AS target_name, n.entity_type AS target_type
                    FROM graph_edges e
                    JOIN graph_nodes n ON e.target_id = n.id
                    WHERE e.user_id = $1 AND e.source_id = $2 AND e.valid_to IS NULL;
                    """,
                    user_id,
                    node.id
                )

                # 3. Fetch active incoming edges
                in_rows = await conn.fetch(
                    """
                    SELECT e.relationship, e.valid_from, n.name AS source_name, n.entity_type AS source_type
                    FROM graph_edges e
                    JOIN graph_nodes n ON e.source_id = n.id
                    WHERE e.user_id = $1 AND e.target_id = $2 AND e.valid_to IS NULL;
                    """,
                    user_id,
                    node.id
                )

                return SubgraphQueryResult(
                    node=node,
                    outgoing_edges=[dict(r) for r in out_rows],
                    incoming_edges=[dict(r) for r in in_rows]
                )
        except Exception as e:
            logger.error(f"Failed to query subgraph: {e}")
            raise StorageOperationError(f"Query subgraph error: {e}")
