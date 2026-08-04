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
from memory.atomic.json_utils import _coerce_json
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


def _is_temporal_query(query_text: str) -> bool:
    """Detect 'before/after/prior to/previously/until' style temporal queries.

    Deterministic keyword check — no LLM, no model. Used to trigger the
    temporal rescue/boost path so time-appropriate facts surface even when
    their vector similarity to the query is low (the 2019 'studied CS' edge).
    """
    q = query_text.lower()
    return any(tok in q for tok in ("before", "after", "prior to", "previously", "until", "since", "earlier", "later"))


def _extract_anchor_date(query_text: str) -> Optional[datetime]:
    """Pull a year (YYYY) or 'YYYY-MM' from a temporal query, if present.

    Returns a UTC datetime anchored at the start of that period, or None.
    Example: 'before founding Kraionyx in 2025' -> 2025-01-01. Used to orient
    the directional temporal boost (boost facts on the correct side of the anchor).
    """
    import re
    m = re.search(r"(19|20)\d{2}(?:-\d{1,2})?", query_text)
    if not m:
        return None
    token = m.group(0)
    try:
        if "-" in token:
            return datetime.fromisoformat(token).replace(tzinfo=timezone.utc)
        return datetime(int(token), 1, 1, tzinfo=timezone.utc)
    except ValueError:
        return None


def _temporal_boost(anchor: Optional[datetime], direction: str, valid_from: Optional[datetime]) -> float:
    """Boost multiplier for a candidate fact under a temporal query.

    direction 'before' -> facts strictly earlier than the anchor get up to +2.0
    (scaled by how much earlier); 'after' -> facts later than the anchor.
    Facts on the wrong side get a mild penalty. When the query has no explicit
    year (anchor None), 'now' is used as the reference so genuinely older/newer
    facts are promoted relative to the rest of the candidate set.
    """
    if valid_from is None:
        return 1.0
    ref = anchor if anchor is not None else datetime.now(timezone.utc)
    vf = valid_from if valid_from.tzinfo else valid_from.replace(tzinfo=timezone.utc)
    delta_days = (vf - ref).total_seconds() / 86400.0
    if direction == "before":
        if delta_days < 0:
            # earlier: stronger boost the further back it is (cap at +2.0)
            return 1.0 + min(2.0, abs(delta_days) / 365.0)
        return 0.5  # wrong side of the anchor
    else:  # after
        if delta_days > 0:
            return 1.0 + min(2.0, delta_days / 365.0)
        return 0.5


def _temporal_direction(query_text: str) -> str:
    q = query_text.lower()
    if any(t in q for t in ("before", "prior to", "previously", "until", "earlier")):
        return "before"
    if any(t in q for t in ("after", "since", "later")):
        return "after"
    return "before"  # default orientation for ambiguous temporal queries


async def _temporal_rescue(
    conn, user_id: str, query_text: str, vector_str: str, anchor: Optional[datetime], direction: str, limit: int
) -> List[Dict[str, Any]]:
    """Rescue time-appropriate facts that vector/FTS similarity dropped below threshold.

    Runs ONLY for temporal queries and ONLY when the candidate set lacks a fact on the
    correct temporal side of the anchor. Pulls active facts whose valid_from is on the
    requested side, ordered by temporal closeness to the anchor, up to `limit` extra rows.
    Deterministic, no LLM. Keeps the change cost-free on the common (non-temporal) path.
    """
    if anchor is None:
        # No explicit year in the query (e.g. "before founding Kraionyx"). Orient by
        # direction only: 'before' -> oldest facts first, 'after' -> newest first.
        if direction == "before":
            order = "valid_from ASC"
            cond = "valid_from IS NOT NULL"
        else:
            order = "valid_from DESC"
            cond = "valid_from IS NOT NULL"
        try:
            rows = await conn.fetch(
                f"""
                SELECT id, user_id, session_id, agent_id, run_id, fact_text, categories,
                       memory_type, metadata, importance_score, belief_confidence,
                       valid_from, valid_to, invalidated_by, provenance_kind,
                       source_episode_id, contradicted_by,
                       (1 - (embedding <=> $1::vector)) AS raw_similarity,
                       (1 - (embedding <=> $1::vector)) * (1.0 / (1.0 + 0.05 * (EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 86400.0))) AS decayed_similarity,
                       ts_rank_cd(to_tsvector('english', fact_text), to_tsquery('english', $2::text)) AS fts_rank,
                       created_at
                FROM atomic_facts
                WHERE user_id = $3 AND is_active = TRUE AND {cond}
                ORDER BY {order}
                LIMIT {limit};
                """,
                vector_str, " | ".join(query_text.split()) or query_text, user_id,
            )
            return list(rows)
        except Exception as e:
            log_error(f"temporal rescue skipped: {e}")
            return []

    if direction == "before":
        cond = "valid_from < $1"
        order = "valid_from DESC"
    else:
        cond = "valid_from > $1"
        order = "valid_from ASC"
    try:
        rows = await conn.fetch(
            f"""
            SELECT id, user_id, session_id, agent_id, run_id, fact_text, categories,
                   memory_type, metadata, importance_score, belief_confidence,
                   valid_from, valid_to, invalidated_by, provenance_kind,
                   source_episode_id, contradicted_by,
                   (1 - (embedding <=> $2::vector)) AS raw_similarity,
                   (1 - (embedding <=> $2::vector)) * (1.0 / (1.0 + 0.05 * (EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 86400.0))) AS decayed_similarity,
                   ts_rank_cd(to_tsvector('english', fact_text), to_tsquery('english', $3::text)) AS fts_rank,
                   created_at
            FROM atomic_facts
            WHERE user_id = $4 AND is_active = TRUE AND {cond}
            ORDER BY {order}
            LIMIT {limit};
            """,
            anchor, vector_str, " | ".join(query_text.split()) or query_text, user_id,
        )
        return list(rows)
    except Exception as e:  # never break the main path
        log_error(f"temporal rescue skipped: {e}")
        return []


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
    # Use the REAL Postgres FTS/BM25 score (ts_rank_cd) instead of a coarse token-overlap
    # heuristic — aligns the fusion with Cognee's bm25_rank in hybrid/ranking.py.
    fts = _minmax([float(getattr(r, "fts_rank", 0.0)) or 0.0 for r in results])

    for r in results:
        fact_lc = r.fact_text.lower()
        graph_boost = 1.0
        if boost_entities and any(ent.lower() in fact_lc for ent in boost_entities):
            graph_boost = 1.25

        base = 0.5 * dense[r.similarity] + 0.25 * rrf[r.rrf_score] + 0.25 * fts[float(getattr(r, "fts_rank", 0.0)) or 0.0]
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
    boost_entities: Optional[List[str]] = None,
    run_id: Optional[str] = None,
    categories: Optional[List[str]] = None,
    memory_type: Optional[str] = None,
    topics: Optional[List[str]] = None,
    themes: Optional[List[str]] = None,
) -> List[FactSearchResult]:
    """Execute Mem0 Multi-Signal RRF (Reciprocal Rank Fusion) hybrid search across Dense Vector, BM25 Full-Text, and Recency Decay, then a deterministic fusion rerank.

    Mem0-equivalent search filters: run_id, categories, memory_type, topics, themes.
    boost_entities: list of activated graph entity names (HippoRAG PPR). Facts that mention
    any boosted entity get a small relevance boost so the knowledge graph actually influences
    what the agent recalls.
    """
    async with measure_latency("memory.atomic.search_facts"):
        try:
            embedding = await embedder.embed_text(query_text)
            vector_str = f"[{','.join(str(x) for x in embedding)}]"

            # BM25/FTS query: OR-join query tokens so ANY term match scores (Cognee/Mem0
            # keyword search rewards partial overlap). Passed as a parameter — never
            # concatenated into SQL — so it is injection-safe.
            fts_query = " | ".join(t for t in query_text.split() if t) or query_text

            async with DatabasePool.acquire() as conn:
                # Build dynamic WHERE for Mem0-equivalent filters.
                # $2 is the OR-joined FTS query (used by to_tsquery); query_text itself is
                # only used in Python for fuzzy matching, so it is NOT a SQL parameter.
                where = ["user_id = $3", "is_active = TRUE"]
                params: List[Any] = [vector_str, fts_query, user_id, session_id, agent_id]
                p = 6
                if session_id is not None:
                    where.append(f"($4::uuid IS NULL OR session_id = $4)")
                else:
                    where.append("($4::uuid IS NULL OR session_id = $4)")
                where.append("($5::text IS NULL OR agent_id = $5)")
                if run_id is not None:
                    where.append(f"run_id = ${p}"); params.append(run_id); p += 1
                if memory_type is not None:
                    where.append(f"memory_type = ${p}"); params.append(memory_type); p += 1
                if categories:
                    where.append(f"categories && ${p}"); params.append(categories); p += 1
                if topics:
                    where.append(f"categories && ${p}"); params.append(topics); p += 1
                if themes:
                    where.append(f"categories && ${p}"); params.append(themes); p += 1

                sql = f"""
                    SELECT
                        id,
                        user_id,
                        session_id,
                        agent_id,
                        run_id,
                        fact_text,
                        categories,
                        memory_type,
                        metadata,
                        importance_score,
                        belief_confidence,
                        valid_from,
                        valid_to,
                        invalidated_by,
                        provenance_kind,
                        source_episode_id,
                        contradicted_by,
                        (1 - (embedding <=> $1::vector)) AS raw_similarity,
                        (1 - (embedding <=> $1::vector)) * (1.0 / (1.0 + 0.05 * (EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 86400.0))) AS decayed_similarity,
                        ts_rank_cd(to_tsvector('english', fact_text), to_tsquery('english', $2::text)) AS fts_rank,
                        created_at
                    FROM atomic_facts
                    WHERE {' AND '.join(where)}
                    ORDER BY embedding <=> $1::vector ASC
                    LIMIT 30;
                """
                rows = await conn.fetch(sql, *params)

                # --- Temporal rescue (zero-LLM, only for temporal queries) ---
                # Vector/FTS similarity alone drops genuinely older/newer facts for
                # "before/after" queries (e.g. the 2019 'studied CS' fact). If no
                # candidate sits on the requested temporal side of the anchor, pull
                # time-appropriate facts so the temporal graph is actually consulted.
                temporal = _is_temporal_query(query_text)
                anchor = _extract_anchor_date(query_text) if temporal else None
                direction = _temporal_direction(query_text) if temporal else "before"
                if temporal:
                    if anchor is not None:
                        side_present = any(
                            (r["valid_from"] is not None)
                            and (
                                (direction == "before" and r["valid_from"] < anchor)
                                or (direction == "after" and r["valid_from"] > anchor)
                            )
                            for r in rows
                        )
                        rescue = not side_present
                    else:
                        # No explicit year: "before/after" asks for temporal context the
                        # vector search dropped. Always rescue the oldest/newest facts so
                        # they can be directionally boosted into the result set.
                        rescue = True
                    if rescue:
                        rescued = await _temporal_rescue(
                            conn, user_id, query_text, vector_str, anchor, direction, limit
                        )
                        seen = {r["id"] for r in rows}
                        for rr in rescued:
                            if rr["id"] not in seen:
                                rows = rows + [rr]
                                seen.add(rr["id"])


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

                    # Temporal directional boost (zero-LLM): for "before/after" queries,
                    # facts on the correct temporal side of the anchor are promoted, and
                    # facts on the wrong side are mildly penalised. No cost on non-temporal path.
                    if temporal:
                        composite_score *= _temporal_boost(anchor, direction, r.get("valid_from"))

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
                            fts_rank=float(r["fts_rank"]),
                            importance_score=float(r["importance_score"]),
                            created_at=r["created_at"],
                            session_id=r["session_id"],
                            agent_id=r["agent_id"],
                            run_id=r.get("run_id"),
                            categories=list(r.get("categories") or []),
                            memory_type=r.get("memory_type", "semantic"),
                            metadata=_coerce_json(r.get("metadata")),
                            belief_confidence=float(r.get("belief_confidence") or 1.0),
                            valid_from=r.get("valid_from"),
                            valid_to=r.get("valid_to"),
                            invalidated_by=list(r.get("invalidated_by") or []),
                            provenance_kind=r.get("provenance_kind", "explicit"),
                            source_episode_id=r.get("source_episode_id"),
                            contradicted_by=list(r.get("contradicted_by") or []),
                        ))
                        # stash composite score for ranking (not part of schema)
                        _composite[fid] = composite_score

                # Sort by composite (importance×retention-decayed RRF) when RRF enabled,
                # otherwise by decayed similarity. RRF score is still returned for inspection.
                if apply_rrf:
                    results.sort(key=lambda x: _composite.get(x.id, x.rrf_score), reverse=True)
                else:
                    results.sort(key=lambda x: x.similarity, reverse=True)

                # Deterministic fusion rerank (always-on, no LLM, no model) — the guaranteed
                # baseline so retrieval quality is never worse than this even if the cross-encoder
                # is unavailable.
                if settings.FACT_RERANK:
                    results = rerank_facts(query_text, results, boost_entities=boost_entities)

                # Primary rerank: Mem0-style local cross-encoder (reference: mem0/mem0/reranker).
                # Improves ordering precision over the fusion score; on any failure it falls back
                # to the fusion ordering (never turns down the query).
                if settings.FACT_RERANKER_ENABLED:
                    try:
                        from memory.atomic.reranker import get_reranker
                        ce = get_reranker()
                        if ce is not None:
                            docs = [
                                {"id": str(r.id), "fact_text": r.fact_text,
                                 "similarity": float(r.similarity),
                                 "rrf_score": float(r.rrf_score),
                                 "importance_score": float(r.importance_score),
                                 "created_at": r.created_at}
                                for r in results
                            ]
                            ranked = ce.rerank(query_text, docs, top_k=limit)
                            by_id = {d["id"]: d for d in ranked}
                            ordered = [by_id[str(r.id)] for r in results if str(r.id) in by_id]
                            if ordered:
                                results = results[:]  # preserve objects
                                # Reorder result objects to match cross-encoder ranking
                                rank_map = {d["id"]: i for i, d in enumerate(ordered)}
                                results.sort(key=lambda x: rank_map.get(str(x.id), 999))
                                for r, d in zip(results, ordered):
                                    r.rrf_score = round(float(d.get("rerank_score", r.rrf_score)), 6)
                    except Exception as e:
                        log_error(f"Cross-encoder rerank skipped (fusion fallback active): {e}")

                # --- Guaranteed temporal slot (zero-LLM) ---
                # For "before/after" queries, ensure the final set contains at least one
                # fact on the requested temporal side. If none made the cut by composite,
                # swap the lowest-composite result for the best rescued candidate that
                # satisfies the direction. This makes the temporal graph authoritative for
                # temporal queries without inflating non-temporal ranking.
                if temporal and results:
                    # Slot is satisfied only if a fact on the correct temporal side is
                    # actually present in the top-N. For no-anchor queries we must compare
                    # against the globally oldest/newest candidate (rescued facts included),
                    # not just among the current results — otherwise the oldest current
                    # result falsely satisfies the slot while the truly-older fact is dropped.
                    all_times = [r.get("valid_from") for r in rows if r.get("valid_from") is not None]
                    global_oldest = min(all_times) if all_times else None
                    global_newest = max(all_times) if all_times else None

                    def _on_side(r):
                        vf = r.valid_from
                        if vf is None:
                            return False
                        if anchor is not None:
                            return (direction == "before" and vf < anchor) or (direction == "after" and vf > anchor)
                        if direction == "before":
                            return global_oldest is not None and vf <= global_oldest
                        return global_newest is not None and vf >= global_newest

                    if not any(_on_side(r) for r in results[:limit]):
                        # The temporally-correct fact may already be in `results` but
                        # ranked beyond `limit` (rescued). Pull the best side fact from
                        # the full `results` list and swap it into the final top-N slot,
                        # replacing the lowest-ranked current result.
                        side_results = [r for r in results if _on_side(r)]
                        if side_results:
                            side_results.sort(
                                key=lambda r: _composite.get(r.id, r.rrf_score), reverse=True
                            )
                            best = side_results[0]
                            top_ids = {r.id for r in results[:limit]}
                            if best.id in top_ids:
                                pass  # already present, nothing to do
                            else:
                                results = results[:limit - 1] + [best] + [
                                    r for r in results[limit:] if r.id != best.id
                                ]

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
