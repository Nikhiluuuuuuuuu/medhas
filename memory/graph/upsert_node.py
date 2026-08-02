"""Layer 4 (Zep/Graphiti): Upsert entity node in temporal graph with dynamic canonicalization."""

import json
from typing import Dict, Any, Optional
from infrastructure.db import DatabasePool
from memory.graph.canonicalize_node import resolve_canonical_node_name
from schemas import GraphNodeSchema
from utils import measure_latency, log_graph, log_error
from core.exceptions import StorageOperationError

from uuid import UUID

async def upsert_node(
    user_id: str,
    name: str,
    entity_type: str,
    attributes: Optional[Dict[str, Any]] = None,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None
) -> GraphNodeSchema:
    """Upsert entity node using dynamic canonical node resolution."""
    attrs = attributes or {}
    canonical_name = await resolve_canonical_node_name(user_id, name)

    async with measure_latency(f"memory.graph.upsert_node ({canonical_name})"):
        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO graph_nodes (user_id, session_id, agent_id, name, entity_type, attributes)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    ON CONFLICT (user_id, name)
                    DO UPDATE SET entity_type = $5, attributes = graph_nodes.attributes || $6::jsonb
                    RETURNING id, user_id, name, entity_type, attributes, created_at;
                    """,
                    user_id,
                    session_id,
                    agent_id,
                    canonical_name,
                    entity_type,
                    json.dumps(attrs)
                )
                assert row is not None, "Failed to upsert node"
                node = GraphNodeSchema(
                    id=row["id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    entity_type=row["entity_type"],
                    attributes=json.loads(row["attributes"]) if isinstance(row["attributes"], str) else dict(row["attributes"]),
                    created_at=row["created_at"]
                )
                return node
        except Exception as e:
            log_error(f"Failed to upsert graph node: {e}")
            raise StorageOperationError(f"Upsert node error: {e}")
