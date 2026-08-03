"""GBrain-inspired typed-link layer over the bi-temporal graph.

GBrain exposes `link` (typed, provenance-tracked), `backlinks`, and `graph`
(depth/direction traversal). Medhas already has bi-temporal edges; this module
adds the typed-link + provenance + traversal surface on top, mirroring GBrain's
link/backlinks/graph-query commands.

All operations are non-destructive: `remove_link` soft-closes the edge (valid_to),
exactly like update_edge, so point-in-time history is preserved.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import UUID
from infrastructure.db import DatabasePool
from schemas import GraphEdgeSchema
from utils import measure_latency, log_graph, log_error
from core.exceptions import StorageOperationError

VALID_LINK_SOURCES = ("manual", "extracted", "inferred")


async def create_link(
    user_id: str,
    source_id: UUID,
    target_id: UUID,
    relationship: str,
    link_type: Optional[str] = None,
    link_source: str = "manual",
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None,
) -> GraphEdgeSchema:
    """Create a typed, provenance-tracked link (GBrain `link`)."""
    if link_source not in VALID_LINK_SOURCES:
        link_source = "manual"
    # Reuse the bi-temporal edge writer (handles contradiction invalidation).
    from memory.graph.update_edge import update_edge
    return await update_edge(
        user_id, source_id, target_id, relationship,
        session_id=session_id, agent_id=agent_id,
        link_type=link_type or relationship, link_source=link_source,
    )


async def remove_link(
    user_id: str,
    source_id: UUID,
    target_id: UUID,
    relationship: Optional[str] = None,
    link_type: Optional[str] = None,
    link_source: Optional[str] = None,
) -> int:
    """Remove (soft-close) matching active links (GBrain `unlink`). Returns count invalidated."""
    async with measure_latency("memory.graph.links.remove_link"):
        try:
            clauses = ["user_id = $1", "source_id = $2", "target_id = $3", "valid_to IS NULL"]
            params: List[Any] = [user_id, source_id, target_id]
            i = 4
            for col, val in (("relationship", relationship), ("link_type", link_type), ("link_source", link_source)):
                if val is not None:
                    params.append(val)
                    clauses.append(f"{col} = ${i}")
                    i += 1
            sql = (
                "UPDATE graph_edges SET valid_to = $" + str(i) + ", expired_at = CURRENT_TIMESTAMP "
                "WHERE " + " AND ".join(clauses)
            )
            params.append(datetime.now(timezone.utc))
            async with DatabasePool.acquire() as conn:
                status = await conn.execute(sql, *params)
                n = int(status.split()[-1]) if status else 0
                return n
        except Exception as e:
            log_error(f"remove_link failed: {e}")
            raise StorageOperationError(f"remove_link error: {e}")


async def get_backlinks(
    user_id: str, node_id: UUID, active_only: bool = True
) -> List[Dict[str, Any]]:
    """Incoming links to a node (GBrain `backlinks`)."""
    async with measure_latency("memory.graph.links.get_backlinks"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT e.id, e.source_id, n.name AS source_name, e.relationship, e.link_type, e.link_source
                    FROM graph_edges e
                    JOIN graph_nodes n ON n.id = e.source_id
                    WHERE e.user_id = $1 AND e.target_id = $2
                      AND ($3::bool IS FALSE OR e.valid_to IS NULL)
                    ORDER BY e.valid_from DESC;
                    """,
                    user_id, node_id, active_only,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            log_error(f"get_backlinks failed: {e}")
            return []


async def traverse_graph(
    user_id: str,
    node_id: UUID,
    direction: str = "both",   # 'out' | 'in' | 'both'
    depth: int = 2,
    link_type: Optional[str] = None,
    active_only: bool = True,
) -> Dict[str, Any]:
    """Breadth-first traversal of the link graph (GBrain `graph` / `graph-query`).

    Returns {'nodes': [...], 'edges': [...]} up to `depth` hops.
    """
    async with measure_latency("memory.graph.links.traverse_graph"):
        try:
            nodes: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []
            frontier = [node_id]
            visited = {node_id}
            async with DatabasePool.acquire() as conn:
                for _ in range(max(1, depth)):
                    next_frontier: List[UUID] = []
                    for nid in frontier:
                        if direction in ("out", "both"):
                            rows = await conn.fetch(
                                """
                                SELECT e.id, e.target_id, t.name AS target_name, e.relationship, e.link_type, e.link_source
                                FROM graph_edges e
                                JOIN graph_nodes t ON t.id = e.target_id
                                WHERE e.user_id = $1 AND e.source_id = $2
                                  AND ($3::bool IS FALSE OR e.valid_to IS NULL)
                                  AND ($4::text IS NULL OR e.link_type = $4);
                                """,
                                user_id, nid, active_only, link_type,
                            )
                            for r in rows:
                                edges.append(dict(r))
                                if r["target_id"] not in visited:
                                    visited.add(r["target_id"])
                                    next_frontier.append(r["target_id"])
                                    nodes[str(r["target_id"])] = {"id": str(r["target_id"]), "name": r["target_name"]}
                        if direction in ("in", "both"):
                            rows = await conn.fetch(
                                """
                                SELECT e.id, e.source_id, s.name AS source_name, e.relationship, e.link_type, e.link_source
                                FROM graph_edges e
                                JOIN graph_nodes s ON s.id = e.source_id
                                WHERE e.user_id = $1 AND e.target_id = $2
                                  AND ($3::bool IS FALSE OR e.valid_to IS NULL)
                                  AND ($4::text IS NULL OR e.link_type = $4);
                                """,
                                user_id, nid, active_only, link_type,
                            )
                            for r in rows:
                                edges.append(dict(r))
                                if r["source_id"] not in visited:
                                    visited.add(r["source_id"])
                                    next_frontier.append(r["source_id"])
                                    nodes[str(r["source_id"])] = {"id": str(r["source_id"]), "name": r["source_name"]}
                    frontier = next_frontier
                    if not frontier:
                        break
            return {"nodes": list(nodes.values()), "edges": edges}
        except Exception as e:
            log_error(f"traverse_graph failed: {e}")
            return {"nodes": [], "edges": []}
