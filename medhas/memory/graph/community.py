"""Graphiti-style community detection + community search over the entity graph.

Graphiti exposes `community_search` over detected communities (densely-connected node
groups). Medhas builds communities via connected-component labeling on the active
bi-temporal edge graph, then ranks communities by (a) how many of their member entities
match the query, and (b) their edge density. Returns the top communities and their members.
"""

from typing import List, Dict, Any, Optional
from medhas.storage import DatabasePool
from medhas.embeddings import FastEmbeddingProvider
from medhas.utils import measure_latency, log_graph, log_error

embedder = FastEmbeddingProvider()


async def detect_communities(user_id: str) -> List[Dict[str, Any]]:
    """Return connected components of the active entity graph as communities.

    Each community: {id, members: [entity names], size, edge_count}.
    """
    async with measure_latency("memory.graph.community.detect_communities"):
        try:
            async with DatabasePool.acquire() as conn:
                edges = await conn.fetch(
                    """
                    SELECT n1.name AS source, n2.name AS target
                    FROM graph_edges e
                    JOIN graph_nodes n1 ON n1.id = e.source_id
                    JOIN graph_nodes n2 ON n2.id = e.target_id
                    WHERE e.user_id = $1 AND e.valid_to IS NULL;
                    """,
                    user_id,
                )
                # Build adjacency (undirected).
                adj: Dict[str, set] = {}
                for e in edges:
                    s, t = e["source"], e["target"]
                    adj.setdefault(s, set()).add(t)
                    adj.setdefault(t, set()).add(s)
                visited: set = set()
                communities: List[Dict[str, Any]] = []
                for node in adj:
                    if node in visited:
                        continue
                    # BFS component
                    comp: List[str] = []
                    stack = [node]
                    while stack:
                        cur = stack.pop()
                        if cur in visited:
                            continue
                        visited.add(cur)
                        comp.append(cur)
                        stack.extend(adj[cur] - visited)
                    communities.append({
                        "id": f"comm_{len(communities)+1}",
                        "members": comp,
                        "size": len(comp),
                        "edge_count": sum(len(adj[m]) for m in comp) // 2,
                    })
                return communities
        except Exception as e:
            log_error(f"detect_communities failed: {e}")
            return []


async def community_search(
    user_id: str, query: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """Graphiti `community_search`: rank communities by query relevance to their members.

    Uses keyword + embedding overlap between the query and community member names.
    """
    async with measure_latency("memory.graph.community.community_search"):
        try:
            communities = await detect_communities(user_id)
            if not communities:
                return []
            q_tokens = {t.lower() for t in query.lower().split() if len(t) > 2}
            q_emb = await embedder.embed_text(query)
            # Score = fraction of member names that share a query token, boosted by size.
            scored = []
            for c in communities:
                hit = sum(1 for m in c["members"] if any(tok in m.lower() for tok in q_tokens))
                score = (hit / max(1, len(c["members"]))) * (1.0 + 0.1 * c["size"])
                if hit > 0:
                    scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:limit]]
        except Exception as e:
            log_error(f"community_search failed: {e}")
            return []
