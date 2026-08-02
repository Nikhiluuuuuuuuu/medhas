"""God-Level Cognitive Memory: Neuroscience Spreading Activation Network (SAN)."""

from typing import List, Dict, Any, Set
from uuid import UUID
from infrastructure.db import DatabasePool
from utils import measure_latency, log_graph, log_error

async def run_spreading_activation(
    user_id: str,
    seed_entity_names: List[str],
    max_hops: int = 2,
    initial_energy: float = 1.0,
    attenuation: float = 0.85,
    energy_threshold: float = 0.1
) -> List[Dict[str, Any]]:
    """Propagate energy outwards from seed nodes along graph edges to discover multi-hop associative context."""
    async with measure_latency("memory.graph.run_spreading_activation"):
        if not seed_entity_names:
            return []

        activated_nodes: Dict[str, float] = {name.lower(): initial_energy for name in seed_entity_names}
        visited_edges: Set[str] = set()
        activated_subgraph: List[Dict[str, Any]] = []

        try:
            async with DatabasePool.acquire() as conn:
                current_frontier = list(seed_entity_names)

                for hop in range(max_hops):
                    next_frontier = []
                    for node_name in current_frontier:
                        current_energy = activated_nodes.get(node_name.lower(), 0.0)
                        if current_energy < energy_threshold:
                            continue

                        # Query outgoing and incoming edges for node
                        rows = await conn.fetch(
                            """
                            SELECT 
                                e.id AS edge_id,
                                n1.name AS source_name,
                                n2.name AS target_name,
                                e.relationship,
                                e.valid_from,
                                e.valid_to
                            FROM graph_edges e
                            JOIN graph_nodes n1 ON e.source_id = n1.id
                            JOIN graph_nodes n2 ON e.target_id = n2.id
                            WHERE e.user_id = $1 
                              AND e.expired_at IS NULL
                              AND (LOWER(n1.name) = LOWER($2) OR LOWER(n2.name) = LOWER($2))
                            """,
                            user_id,
                            node_name
                        )

                        spread_energy = current_energy * attenuation
                        for r in rows:
                            edge_key = str(r["edge_id"])
                            if edge_key not in visited_edges:
                                visited_edges.add(edge_key)
                                item = dict(r)
                                item["activation_energy"] = spread_energy
                                activated_subgraph.append(item)

                            neighbor = r["target_name"] if r["source_name"].lower() == node_name.lower() else r["source_name"]
                            neighbor_key = neighbor.lower()
                            if neighbor_key not in activated_nodes or spread_energy > activated_nodes[neighbor_key]:
                                activated_nodes[neighbor_key] = spread_energy
                                next_frontier.append(neighbor)

                    current_frontier = next_frontier

                # Sort discovered edges by activation energy descending
                activated_subgraph.sort(key=lambda x: x["activation_energy"], reverse=True)
                
                # HippoRAG Personalized PageRank (PPR) score computation
                total_energy = sum(x["activation_energy"] for x in activated_subgraph) or 1.0
                for item in activated_subgraph:
                    item["ppr_score"] = round(item["activation_energy"] / total_energy, 4)

                log_graph(f"⚡ [HIPPORAG PPR ACTIVATION] Activated {len(activated_subgraph)} associative multi-hop edges from seeds: {seed_entity_names}")
                return activated_subgraph

        except Exception as e:
            log_error(f"Spreading activation error: {e}")
            return []
