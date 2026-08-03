"""Layer 3 (Mem0 & GBrain): Search atomic facts using RRF (Reciprocal Rank Fusion) hybrid search & recency decay."""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timezone
from infrastructure.db import DatabasePool
from infrastructure.llm import FastEmbeddingProvider
from schemas import FactSearchResult
from config import settings
from utils import measure_latency, log_atomic, log_error
from core.exceptions import StorageOperationError
from memory.atomic.ebbinghaus_decay import calculate_ebbinghaus_retention
embedder = FastEmbeddingProvider()


def _minmax(values: List[float]) -> Dict[float, float]:
    """Map each value to 0..1 by min-max; constant inputs map to 1.0."""
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {v: 1.0 for v in values}
    return {v: (v - lo) / (hi - lo) for v in values}


def rerank_facts(
    query_text: str,
    results: List[FactSearchResult],
    boost_entities: Optional[List[str]] = None,
) -> List[FactSearchResult]:
    """Deterministic multi-signal fusion reranker (no LLM, no network — always on, never turns down).

    Combines normalized Dense similarity, BM25/keyword match, RRF rank, Ebbinghaus recency
    decay, importance weight, and HippoRAG graph-boost into a single 0..1 relevance score.
    This is the Mem0 rerank step done deterministically (Mem0 uses an LLM reranker; we use a
    stable fusion score so ranking cannot regress due to an LLM misorder and adds zero latency).
    """
    if not results:
        return results

    dense = _minmax([float(r.similarity) for r in results])
    rrf = _minmax([float(r.rrf_score) for r in results])
    q_tokens = {t.lower() for t in query_text.split() if len(t) > 2}

    for r in results:
        fact_lc = r.fact_text.lower()
        kw_overlap = sum(1 for t in q_tokens if t in fact_lc)
        kw_ratio = kw_overlap / max(1, len(q_tokens))

        graph_boost = 1.0
        if boost_entities and any(ent.lower() in fact_lc for ent in boost_entities):
            graph_boost = 1.25

        base = 0.5 * dense[r.similarity] + 0.25 * rrf[r.rrf_score] + 0.25 * kw_ratio
        imp = max(0.1, min(1.0, float(r.importance_score) / 10.0))
        retention = calculate_ebbinghaus_retention(r.created_at)
        score = base * imp * retention * graph_boost
        r.rrf_score = round(score, 6)

    results.sort(key=lambda x: x.rrf_score, reverse=True)
    return results


async def search_facts(
    user_id: str,
    query_text: str,
    limit: int = settings.TOP_K_FACTS,
    similarity_threshold: float = settings.FACT_SIMILARITY_THRESHOLD,
    apply_rrf: bool = True,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None,
    boost_entities: Optional[List[str]] = None
) -> List[FactSearchResult]:
    """Execute Mem0 Multi-Signal RRF (Reciprocal Rank Fusion) hybrid search across Dense Vector, BM25 Full-Text, and Recency Decay, then a deterministic fusion rerank.

    boost_entities: list of activated graph entity names (HippoRAG PPR). Facts that mention
    any boosted entity get a small relevance boost so the knowledge graph actually influences
    what the agent recalls.
    """
    async with measure_latency("memory.atomic.search_facts"):
        try:
            embedding = await embedder.embed_text(query_text)
            vector_str = f"[{','.join(str(x) for x in embedding)}]"

            async with DatabasePool.acquire() as conn:
                # 1. Fetch vector similarity & BM25 rank candidates with multi-scope filtering
                rows = await conn.fetch(
                    """
                    SELECT 
                        id, 
                        user_id,
                        session_id,
                        agent_id,
                        fact_text, 
                        importance_score,
                        (1 - (embedding <=> $1::vector)) AS raw_similarity,
                        (1 - (embedding <=> $1::vector)) * (1.0 / (1.0 + 0.05 * (EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 86400.0))) AS decayed_similarity,
                        ts_rank_cd(to_tsvector('english', fact_text), plainto_tsquery('english', $2)) AS fts_rank,
                        created_at
                    FROM atomic_facts
                    WHERE user_id = $3 
                      AND is_active = TRUE
                      AND ($4::uuid IS NULL OR session_id = $4)
                      AND ($5::text IS NULL OR agent_id = $5)
                    ORDER BY embedding <=> $1::vector ASC
                    LIMIT 30;
                    """,
                    vector_str,
                    query_text,
                    user_id,
                    session_id,
                    agent_id
                )

                if not rows:
                    return []

                # Rank list 1: Vector similarity rank (decayed)
                vector_sorted = sorted(rows, key=lambda x: float(x["decayed_similarity"]), reverse=True)
                vector_ranks: Dict[UUID, int] = {r["id"]: i + 1 for i, r in enumerate(vector_sorted)}

                # Rank list 2: Full-Text Search & Typo-Tolerant Token match rank
                query_tokens = [t.lower() for t in query_text.split() if len(t) > 2]

                def calculate_fuzzy_match_score(fact: str) -> float:
                    fact_lower = fact.lower()
                    score = 0.0
                    for tok in query_tokens:
                        if tok in fact_lower:
                            score += 1.0
                        else:
                            # Typo tolerance check (e.g., 'techcrop' vs 'techcorp')
                            for word in fact_lower.split():
                                clean_word = word.strip(".,!?:;\"'()[]")
                                if len(tok) >= 4 and len(clean_word) >= 4 and abs(len(tok) - len(clean_word)) <= 2:
                                    # Count matching character overlaps
                                    common = sum(min(tok.count(c), clean_word.count(c)) for c in set(tok))
                                    if common >= len(tok) - 2:
                                        score += 0.8
                                        break
                    return score

                fts_sorted = sorted(
                    rows,
                    key=lambda r: float(r["fts_rank"]) + calculate_fuzzy_match_score(r["fact_text"]),
                    reverse=True
                )
                keyword_ranks: Dict[UUID, int] = {r["id"]: i + 1 for i, r in enumerate(fts_sorted)}

                # Calculate Multi-Signal RRF score: 1 / (60 + r1) + 1 / (60 + r2)
                # Then apply Mem0/Cognee-inspired importance × Ebbinghaus recency
                # decay so that high-value, recently-reinforced facts rank higher.
                results: List[FactSearchResult] = []
                _composite: Dict[UUID, float] = {}
                for r in rows:
                    fid = r["id"]
                    r_vec = vector_ranks.get(fid, 100)
                    r_kw = keyword_ranks.get(fid, 100)

                    rrf_val = (1.0 / (60.0 + r_vec)) + (1.0 / (60.0 + r_kw))
                    sim = float(r["decayed_similarity"])
                    fuzzy_score = calculate_fuzzy_match_score(r["fact_text"])

                    # Importance weight (1..10 -> 0.1..1.0) × Ebbinghaus retention (0..1)
                    importance_norm = max(0.1, min(1.0, float(r["importance_score"]) / 10.0))
                    retention = calculate_ebbinghaus_retention(
                        r["created_at"] if isinstance(r["created_at"], datetime)
                        else datetime.now(timezone.utc)
                    )
                    composite_score = rrf_val * importance_norm * retention

                    # HippoRAG PPR boost: facts mentioning an activated graph entity rank higher
                    if boost_entities:
                        fact_lc = r["fact_text"].lower()
                        if any(ent.lower() in fact_lc for ent in boost_entities):
                            composite_score *= 1.25

                    # Include if vector similarity matches, FTS matches, or fuzzy typo-tolerant match succeeds
                    if sim >= 0.40 or float(r["fts_rank"]) > 0.1 or fuzzy_score > 0.5:
                        results.append(FactSearchResult(
                            id=fid,
                            fact_text=r["fact_text"],
                            similarity=sim,
                            rrf_score=rrf_val,
                            importance_score=float(r["importance_score"]),
                            created_at=r["created_at"],
                            session_id=r["session_id"],
                            agent_id=r["agent_id"]
                        ))
                        # stash composite score for ranking (not part of schema)
                        _composite[fid] = composite_score

                # Sort by composite (importance×retention-decayed RRF) when RRF enabled,
                # otherwise by decayed similarity. RRF score is still returned for inspection.
                if apply_rrf:
                    results.sort(key=lambda x: _composite.get(x.id, x.rrf_score), reverse=True)
                else:
                    results.sort(key=lambda x: x.similarity, reverse=True)

                # Deterministic fusion rerank (closes the Mem0 rerank gap; always-on, no LLM).
                if settings.FACT_RERANK:
                    results = rerank_facts(query_text, results, boost_entities=boost_entities)

                return results[:limit]

        except Exception as e:
            log_error(f"Failed to search facts: {e}")
            raise StorageOperationError(f"Vector search error: {e}")

async def search_facts_dual_level(
    user_id: str,
    query_text: str,
    limit: int = settings.TOP_K_FACTS,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None
) -> Dict[str, Any]:
    """LightRAG Dual-Level Retrieval: Parallel extraction of Low-Level (exact facts/triplets) and High-Level (abstract themes/graph concepts)."""
    async with measure_latency("memory.atomic.search_facts_dual_level"):
        try:
            # 1. Low-Level Retrieval: Specific detailed facts
            low_level_facts = await search_facts(
                user_id=user_id,
                query_text=query_text,
                limit=limit,
                session_id=session_id,
                agent_id=agent_id
            )
            if settings.FACT_RERANK:
                low_level_facts = rerank_facts(query_text, low_level_facts)

            # 2. High-Level Retrieval: Broad subgraph concept relationships and reflections
            async with DatabasePool.acquire() as conn:
                reflections = await conn.fetch(
                    """
                    SELECT fact_text, importance_score, created_at
                    FROM atomic_facts
                    WHERE user_id = $1 AND is_active = TRUE AND fact_text LIKE '[Reflection]%'
                    ORDER BY created_at DESC
                    LIMIT 5;
                    """,
                    user_id
                )

                graph_summary = await conn.fetch(
                    """
                    SELECT n.name, n.entity_type, e.relationship, t.name as target_name
                    FROM graph_nodes n
                    JOIN graph_edges e ON n.id = e.source_id
                    JOIN graph_nodes t ON e.target_id = t.id
                    WHERE n.user_id = $1 AND e.valid_to IS NULL
                    LIMIT 10;
                    """,
                    user_id
                )

            return {
                "low_level_facts": [f.model_dump() for f in low_level_facts],
                "high_level_concepts": {
                    "reflections": [dict(r) for r in reflections],
                    "subgraph_relationships": [dict(g) for g in graph_summary]
                }
            }
        except Exception as e:
            log_error(f"Dual-level search error: {e}")
            return {"low_level_facts": [], "high_level_concepts": {}}

async def get_all_active_facts(user_id: str) -> List[Dict[str, Any]]:
    """Retrieve all active atomic facts without threshold filtering for audit verification."""
    async with measure_latency("memory.atomic.get_all_active_facts"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, fact_text, importance_score, created_at
                    FROM atomic_facts
                    WHERE user_id = $1 AND is_active = TRUE
                    ORDER BY created_at ASC;
                    """,
                    user_id
                )
                return [dict(r) for r in rows]
        except Exception as e:
            log_error(f"Error fetching all active facts: {e}")
            return []
