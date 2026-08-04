"""H2 — Open anaphora / coreference resolution for queries (and fact seeding).

Resolves vague references (pronouns, "the company that X", "it/they") to concrete
graph entities using the LLM (agi.llm_extract.resolve_entities_open), grounded against
the actual entities present in the user's memory graph. No closed keyword lists — the
model decides what each reference maps to. Falls back to capitalized-token extraction
if the LLM is unavailable.

This makes multi-hop recall work on natural questions like "who advised the founder of
the startup that makes Lumina" or "what did he prefer", with no hardcoded patterns.
"""

from typing import List, Optional
from medhas.storage import DatabasePool
from medhas.utils import measure_latency, log_error

from medhas.extraction.entities import query_entities
from medhas.extraction.llm_extract import resolve_entities_open


async def _known_entities(user_id: str) -> List[str]:
    """All entity names currently in the user's graph (for LLM grounding)."""
    async with DatabasePool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT name FROM graph_nodes WHERE user_id = $1 ORDER BY created_at DESC;",
            user_id,
        )
        return [r["name"] for r in rows]


async def _graph_neighbors(user_id: str, entity: str) -> List[str]:
    """1-hop neighbor entity names of `entity` in the graph (for anaphora bridging)."""
    async with DatabasePool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT n.name
            FROM graph_nodes n
            JOIN graph_edges e ON (e.source_id = n.id OR e.target_id = n.id)
            JOIN graph_nodes sn ON (sn.id = e.source_id OR sn.id = e.target_id)
            WHERE (n.user_id = $1) AND sn.name = $2 AND n.name <> $2
            LIMIT 10;
            """,
            user_id, entity,
        )
        return [r["name"] for r in rows]


async def _nearest_entities_by_embedding(user_id: str, query: str, k: int = 12) -> List[str]:
    """ATOM-style adaptive grounding: return the top-k graph entities whose embedding is
    nearest to the query, instead of the entire entity list.

    Graphiti/Zep prompt the LLM with ALL previous entities, which degrades as the graph
    scales to millions of nodes (ATOM, ACL 2026 Findings). We instead retrieve the k
    entities most semantically relevant to the query via local embeddings, so the resolver
    only sees a small, relevant candidate set regardless of graph size. No network — the
    FastEmbed model already loaded for search/reranking is reused.
    """
    try:
        from medhas.embeddings import FastEmbeddingProvider
        qvec = await FastEmbeddingProvider().embed_text(query)
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT name, embedding <-> $2::vector AS dist
                FROM graph_nodes
                WHERE user_id = $1 AND embedding IS NOT NULL
                ORDER BY dist ASC
                LIMIT $3;
                """,
                user_id, f"[{','.join(str(x) for x in qvec)}]", k,
            )
            return [r["name"] for r in rows]
    except Exception as e:
        log_error(f"nearest-entity retrieval failed, falling back to full list: {e}")
        return await _known_entities(user_id)


async def resolve_query_entities(query: str, user_id: str) -> List[str]:
    """Return concrete entity names for a query, with references resolved via the graph.

    Adaptive resolution (ATOM-style): the LLM resolver is grounded against the top-k
    embedding-nearest entities (not the whole graph), so quality holds as the graph grows.
    Best-effort: on any failure returns the plain capitalized entities. When resolution
    yields nothing and the query is anaphoric (he/she/they/it), falls back to the most
    salient person in the graph so pronoun questions still resolve without the LLM.

    Graph bridging: if the query mentions a known entity (e.g. "KareOS"), its 1-hop
    neighbors are also returned, so descriptions like "the company that builds KareOS"
    resolve to the builder (e.g. "Kraionyx AI") without a keyword dictionary.
    """
    async with measure_latency("agi.anaphora.resolve_query_entities"):
        try:
            # ATOM adaptive grounding: nearest-k by embedding, not the full entity set
            known = await _nearest_entities_by_embedding(user_id, query, k=12)
            if not known:
                known = await _known_entities(user_id)
            if known:
                resolved = await resolve_entities_open(query, known)
                # keep original (capitalized) mentions too, de-duplicated case-insensitively
                seen = {r.lower() for r in resolved}
                for e in query_entities(query):
                    if e.lower() not in seen:
                        resolved.append(e); seen.add(e.lower())
                # graph bridge: any known entity named in the query pulls in its neighbors
                q_lower = query.lower()
                for ent in known:
                    if ent.lower() in q_lower:
                        for nb in await _graph_neighbors(user_id, ent):
                            if nb.lower() not in seen:
                                resolved.append(nb); seen.add(nb.lower())
                if resolved:
                    return resolved
            # empty resolution + anaphoric query -> most salient person (pronoun fallback)
            if any(p in query.lower().split() for p in ("he", "she", "they", "it")):
                person = await _most_salient_person(user_id)
                if person:
                    return [person]
            return query_entities(query)
        except Exception as e:
            log_error(f"resolve_query_entities skipped: {e}")
            return query_entities(query)


async def _most_salient_person(user_id: str) -> Optional[str]:
    """Open heuristic for pronoun resolution ('he'/'she' -> the user's most prominent person).

    Picks the PERSON node with the highest graph degree (most connections), which is the
    natural candidate for an unqualified 'he'/'she'. No keyword/pronoun lists — just graph
    centrality over typed nodes. Returns None if no PERSON node exists.
    """
    async with DatabasePool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT n.name
            FROM graph_nodes n
            LEFT JOIN graph_edges e ON (e.source_id = n.id OR e.target_id = n.id)
            WHERE n.user_id = $1 AND n.entity_type = 'PERSON'
            GROUP BY n.id, n.name
            ORDER BY COUNT(e.id) DESC, n.created_at DESC
            LIMIT 1;
            """,
            user_id,
        )
        return row["name"] if row else None


__all__ = ["resolve_query_entities", "_most_salient_person"]
