"""Layer 4 (Zep/Graphiti): Insert or update bi-temporal relationship edge."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID
from infrastructure.db import DatabasePool
from schemas import GraphEdgeSchema
from utils import measure_latency, log_graph, log_error
from core.exceptions import StorageOperationError

async def update_edge(
    user_id: str,
    source_id: UUID,
    target_id: UUID,
    relationship: str,
    valid_from: Optional[datetime] = None,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None
) -> GraphEdgeSchema:
    """Update bi-temporal relationship: set valid_to on old edges and insert new active edge."""
    vf = valid_from or datetime.now(timezone.utc)
    async with measure_latency(f"memory.graph.update_edge ({relationship})"):
        try:
            async with DatabasePool.acquire() as conn:
                # 1. Invalidate any existing active edges between source and target or for the same relationship type
                await conn.execute(
                    """
                    UPDATE graph_edges
                    SET valid_to = $4, expired_at = CURRENT_TIMESTAMP
                    WHERE user_id = $1 AND source_id = $2 AND (target_id = $3 OR relationship = $5) AND valid_to IS NULL;
                    """,
                    user_id,
                    source_id,
                    target_id,
                    vf,
                    relationship
                )


                # 2. Insert new bi-temporal edge (valid_to = NULL means currently valid)
                row = await conn.fetchrow(
                    """
                    INSERT INTO graph_edges (user_id, session_id, agent_id, source_id, target_id, relationship, valid_from, valid_to)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, NULL)
                    RETURNING id, user_id, source_id, target_id, relationship, valid_from, valid_to, created_at;
                    """,
                    user_id,
                    session_id,
                    agent_id,
                    source_id,
                    target_id,
                    relationship,
                    vf
                )
                assert row is not None, "Failed to insert bi-temporal edge"
                edge = GraphEdgeSchema(
                    id=row["id"],
                    user_id=row["user_id"],
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relationship=row["relationship"],
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                    created_at=row["created_at"]
                )
                log_graph(f"Bi-temporal edge shift: [bold white]{source_id}[/bold white] --[[cyan]{relationship}[/cyan]]--> [bold white]{target_id}[/bold white]")
                return edge
        except Exception as e:
            log_error(f"Failed to update graph edge: {e}")
            raise StorageOperationError(f"Update edge error: {e}")
