"""Layer 4 (Cognee): Export Knowledge Graph in D3.js and NetworkX JSON formats."""

from typing import Dict, Any, List
from medhas.storage import DatabasePool
from medhas.utils import measure_latency, log_graph, log_error

async def export_knowledge_graph(user_id: str) -> Dict[str, Any]:
    """Export user's temporal knowledge graph to D3.js and NetworkX compatible JSON format."""
    async with measure_latency("memory.graph.export_knowledge_graph"):
        try:
            async with DatabasePool.acquire() as conn:
                nodes_rows = await conn.fetch(
                    """
                    SELECT id, name, entity_type, attributes, created_at
                    FROM graph_nodes
                    WHERE user_id = $1;
                    """,
                    user_id
                )

                edges_rows = await conn.fetch(
                    """
                    SELECT id, source_id, target_id, relationship, valid_from, valid_to, created_at
                    FROM graph_edges
                    WHERE user_id = $1 AND valid_to IS NULL;
                    """,
                    user_id
                )

                nodes: List[Dict[str, Any]] = [
                    {
                        "id": str(n["id"]),
                        "name": n["name"],
                        "entity_type": n["entity_type"],
                        "created_at": n["created_at"].isoformat() if n["created_at"] else None
                    }
                    for n in nodes_rows
                ]

                valid_node_ids = {n["id"] for n in nodes}
                links: List[Dict[str, Any]] = [
                    {
                        "id": str(e["id"]),
                        "source": str(e["source_id"]),
                        "target": str(e["target_id"]),
                        "relationship": e["relationship"],
                        "valid_from": e["valid_from"].isoformat() if e["valid_from"] else None
                    }
                    for e in edges_rows
                    if str(e["source_id"]) in valid_node_ids and str(e["target_id"]) in valid_node_ids
                ]


                graph_export = {
                    "user_id": user_id,
                    "directed": True,
                    "multigraph": False,
                    "nodes": nodes,
                    "links": links,
                    "stats": {
                        "total_nodes": len(nodes),
                        "total_active_edges": len(links)
                    }
                }
                log_graph(f"🕸️ [COGNEE EXPORT] Exported knowledge graph: {len(nodes)} nodes, {len(links)} active edges")
                return graph_export
        except Exception as e:
            log_error(f"Failed to export knowledge graph: {e}")
            return {"user_id": user_id, "nodes": [], "links": [], "error": str(e)}
