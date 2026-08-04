"""Layer 4 (Zep/Graphiti): Upsert entity node in temporal graph with dynamic canonicalization + semantic merge."""

import json
from typing import Dict, Any, Optional
from medhas.storage import DatabasePool
from medhas.embeddings import FastEmbeddingProvider
from medhas.memory.graph.canonicalize_node import resolve_canonical_node_name
from medhas.schemas import GraphNodeSchema
from medhas.utils import measure_latency, log_graph, log_error
from medhas.core.exceptions import StorageOperationError

from uuid import UUID

embedder = FastEmbeddingProvider()

SEMANTIC_MERGE_THRESHOLD = 0.95  # Graphiti main.py:1123 — exact OR semantic(>=0.95) match


async def _semantic_match_node(user_id: str, name: str) -> Optional[str]:
    """Find an existing node whose embedding is >=0.95 cosine similar (Graphiti-style merge)."""
    try:
        emb = await embedder.embed_text(name)
        emb_str = f"[{','.join(str(x) for x in emb)}]"
        async with DatabasePool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT name, 1 - (embedding <=> $2::vector) AS sim
                FROM graph_nodes
                WHERE user_id = $1 AND embedding IS NOT NULL
                ORDER BY embedding <=> $2::vector ASC
                LIMIT 1;
                """,
                user_id, emb_str,
            )
            if row and float(row["sim"]) >= SEMANTIC_MERGE_THRESHOLD:
                return row["name"]
    except Exception as e:
        log_error(f"Semantic node match skipped: {e}")
    return None


async def upsert_node(
    user_id: str,
    name: str,
    entity_type: str,
    attributes: Optional[Dict[str, Any]] = None,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None
) -> GraphNodeSchema:
    """Upsert entity node using dynamic canonical node resolution + semantic merge (>=0.95)."""
    attrs = attributes or {}
    # 1. Case/space-insensitive canonicalization (BUG 2 fix)
    canonical_name = await resolve_canonical_node_name(user_id, name)
    # 2. Semantic merge: if a near-identical node exists by embedding, reuse it (Mem0/Cognee entity resolution)
    if canonical_name == name.strip():
        semantic_name = await _semantic_match_node(user_id, name)
        if semantic_name:
            canonical_name = semantic_name
            log_graph(f"Semantic-merged entity '{name}' -> '{canonical_name}' (cosine>=0.95)")

    async with measure_latency(f"memory.graph.upsert_node ({canonical_name})"):
        try:
            # Persist embedding for future semantic merges
            try:
                node_emb = await embedder.embed_text(canonical_name)
                node_emb_str = f"[{','.join(str(x) for x in node_emb)}]"
            except Exception:
                node_emb_str = None

            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO graph_nodes (user_id, session_id, agent_id, name, entity_type, attributes, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::vector)
                    ON CONFLICT (user_id, name)
                    DO UPDATE SET entity_type = $5, attributes = graph_nodes.attributes || $6::jsonb
                    RETURNING id, user_id, name, entity_type, attributes, created_at;
                    """,
                    user_id,
                    session_id,
                    agent_id,
                    canonical_name,
                    entity_type,
                    json.dumps(attrs),
                    node_emb_str,
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
