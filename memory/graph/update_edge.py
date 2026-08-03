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
    agent_id: Optional[str] = None,
    link_type: Optional[str] = None,
    link_source: str = "extracted"
) -> GraphEdgeSchema:
    """Update bi-temporal relationship: set valid_to on old edges and insert new active edge."""
    vf = valid_from or datetime.now(timezone.utc)
    link_type = link_type or relationship
    async with measure_latency(f"memory.graph.update_edge ({relationship})"):
        try:
            async with DatabasePool.acquire() as conn:
                # 0. Idempotency / dedup (Cognee merge_deduplicated_edges semantics): if an
                # ACTIVE edge with the exact same (source, target, relationship) already exists,
                # reuse it instead of creating a duplicate active edge. This mirrors Cognee's
                # identity-based edge dedup and prevents unintended duplicate edges in the graph.
                existing_same = await conn.fetchrow(
                    """
                    SELECT id FROM graph_edges
                    WHERE user_id = $1 AND source_id = $2 AND target_id = $3
                      AND relationship = $4 AND valid_to IS NULL
                    LIMIT 1;
                    """,
                    user_id, source_id, target_id, relationship,
                )
                if existing_same is not None:
                    log_graph(
                        f"Edge dedup: reused active edge [bold white]{source_id}[/bold white] "
                        f"--[[cyan]{relationship}[/cyan]]--> [bold white]{target_id}[/bold white]"
                    )
                    row = await conn.fetchrow(
                        """
                        SELECT id, user_id, source_id, target_id, relationship, link_type,
                               link_source, valid_from, valid_to, created_at
                        FROM graph_edges WHERE id = $1;
                        """,
                        existing_same["id"],
                    )
                    return GraphEdgeSchema(
                        id=row["id"], user_id=row["user_id"], source_id=row["source_id"],
                        target_id=row["target_id"], relationship=row["relationship"],
                        link_type=row["link_type"], link_source=row["link_source"],
                        valid_from=row["valid_from"], valid_to=row["valid_to"],
                        created_at=row["created_at"],
                    )

                # 1. Invalidate the prior active edge that this new edge contradicts:
                # same subject (source) + same relationship, but a DIFFERENT target. This is the
                # Graphiti bi-temporal pattern — the old claim is soft-closed (valid_to set),
                # preserving point-in-time history, while multi-valued relationships to other
                # targets are left intact.
                await conn.execute(
                    """
                    UPDATE graph_edges
                    SET valid_to = $4, expired_at = CURRENT_TIMESTAMP
                    WHERE user_id = $1 AND source_id = $2 AND relationship = $5
                      AND target_id <> $3 AND valid_to IS NULL;
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
                    INSERT INTO graph_edges (user_id, session_id, agent_id, source_id, target_id, relationship, link_type, link_source, valid_from, valid_to)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NULL)
                    RETURNING id, user_id, source_id, target_id, relationship, link_type, link_source, valid_from, valid_to, created_at;
                    """,
                    user_id,
                    session_id,
                    agent_id,
                    source_id,
                    target_id,
                    relationship,
                    link_type,
                    link_source,
                    vf
                )
                assert row is not None, "Failed to insert bi-temporal edge"
                edge = GraphEdgeSchema(
                    id=row["id"],
                    user_id=row["user_id"],
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relationship=row["relationship"],
                    link_type=row["link_type"],
                    link_source=row["link_source"],
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                    created_at=row["created_at"]
                )
                log_graph(f"Bi-temporal edge shift: [bold white]{source_id}[/bold white] --[[cyan]{relationship}[/cyan]]--> [bold white]{target_id}[/bold white]")
                return edge
        except Exception as e:
            log_error(f"Failed to update graph edge: {e}")
            raise StorageOperationError(f"Update edge error: {e}")
