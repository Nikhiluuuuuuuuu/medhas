"""G2 — Build / extend the entity graph from a stored fact.

Wires the existing graph substrate (memory.graph.upsert_node, memory.graph.update_edge)
into the AGI engine's write path so that multi-hop recall becomes possible. This is a
thin composer: it extracts entities + relations (agi.entities) and writes bi-temporal
edges. It is best-effort — any failure is logged, never raised, so a graph issue can
never break `remember`.

References: Zep/Graphiti entity + bi-temporal edge model (E10/E21).
"""

from typing import Optional
from uuid import UUID

from medhas.storage import DatabasePool
from medhas.utils import measure_latency, log_graph, log_error
from medhas.extraction.entities import extract_entities, query_entities
from medhas.extraction.llm_extract import extract_graph_open


async def build_fact_graph(
    user_id: str,
    fact_id: UUID,
    fact_text: str,
    *,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None,
) -> int:
    """Extract entities/relations from a fact and upsert them into the graph.

    Uses LLM-based open extraction (agi.llm_extract) so new relations, lowercase
    names, and novel phrasings are handled without a closed keyword vocabulary.
    Falls back to the dependency-free heuristic extractor on LLM failure.
    Returns the number of edges created. Best-effort.
    """
    async with measure_latency("agi.graph_build.build_fact_graph"):
        try:
            from medhas.memory.graph.upsert_node import upsert_node
            from medhas.memory.graph.update_edge import update_edge

            relations, entity_hints = await extract_graph_open(fact_text, user_id=user_id)
            type_by_name = {h["name"].lower(): h["type"] for h in entity_hints}
            if not relations:
                return 0

            edges = 0
            for subj, rel, obj in relations:
                if subj.lower() == obj.lower():
                    continue  # never create self-loop edges
                subj_type = type_by_name.get(subj.lower(), "ENTITY")
                subj_node = await upsert_node(user_id, subj, entity_type=subj_type)
                obj_type = type_by_name.get(obj.lower(), "ENTITY")
                obj_node = await upsert_node(user_id, obj, entity_type=obj_type)
                # link this fact to both endpoint nodes so multi-hop recall can
                # resolve facts via the graph even when fact_text is later augmented.
                await _link_fact_to_node(user_id, subj_node.id, fact_id)
                await _link_fact_to_node(user_id, obj_node.id, fact_id)
                await update_edge(
                    user_id,
                    subj_node.id,
                    obj_node.id,
                    relationship=rel,
                    session_id=session_id,
                    agent_id=agent_id,
                    link_type=rel,
                    link_source="extracted",
                )
                edges += 1
            if edges:
                log_graph(f"G2 graph build: +{edges} edges from fact {fact_id}")
            return edges
        except Exception as e:
            log_error(f"graph_build skipped (non-fatal): {e}")
            return 0


async def relate_facts(
    user_id: str,
    source_fact_id: UUID,
    target_fact_id: UUID,
    relationship: str = "RELATED_TO",
) -> None:
    """Optional explicit link between two fact rows (for contradiction / evolution wiring)."""
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                """UPDATE atomic_facts SET linked_ids =
                       array_append(COALESCE(linked_ids, ARRAY[]::uuid[]), $2)
                   WHERE id = $1 AND NOT ($2 = ANY(COALESCE(linked_ids, ARRAY[]::uuid[])));""",
                source_fact_id, target_fact_id,
            )
    except Exception as e:
        log_error(f"relate_facts skipped: {e}")


async def _link_fact_to_node(user_id: str, node_id: UUID, fact_id: UUID) -> None:
    """Record that `fact_id` mentions `node_id` (appended to graph_nodes.fact_ids).

    This fact<->node association lets multi-hop recall resolve facts via the graph even
    after engine.remember augments/rewrites fact_text (which would otherwise break a
    naive ILIKE-on-fact_text match)."""
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                """
                UPDATE graph_nodes
                SET fact_ids = array_append(COALESCE(fact_ids, ARRAY[]::uuid[]), $3)
                WHERE id = $2 AND user_id = $1
                  AND NOT ($3 = ANY(COALESCE(fact_ids, ARRAY[]::uuid[])));
                """,
                user_id, node_id, fact_id,
            )
    except Exception as e:
        log_error(f"_link_fact_to_node skipped: {e}")


__all__ = ["build_fact_graph", "relate_facts"]
