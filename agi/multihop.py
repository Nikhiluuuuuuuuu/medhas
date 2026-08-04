"""G2 — Multi-hop recall via the entity graph.

For a query, extract mentioned entities, pull each entity's active subgraph
(query_subgraph), then follow one more hop (2-hop) to collect related facts that a
pure vector search would miss — e.g. "who co-founded the company that builds KareOS":
  hop1: KareOS ->[BUILDS]-> Kraionyx AI
  hop2: Kraionyx AI ->[CO_FOUNDED]-> Nikhil
Returns candidate fact rows (so the existing reranker/abstention layer can score them).
"""

from typing import List, Dict, Any, Sequence, Optional, Set
from datetime import datetime, timezone

from infrastructure.db import DatabasePool
from utils import measure_latency, log_atomic, log_error

from agi.entities import query_entities
from agi.metamemory import feeling_of_knowing
from agi.llm_extract import resolve_entities_open


async def _facts_for_entity(user_id: str, entity_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Return active facts associated with an entity.

    Primary path: follow the graph node's fact_ids link (populated by build_fact_graph),
    which is immune to fact_text augmentation. Fallback: distinctive-token ILIKE on
    fact_text (for entities never linked, or partially augmented text).
    """
    async with DatabasePool.acquire() as conn:
        node = await conn.fetchrow(
            "SELECT id, fact_ids FROM graph_nodes WHERE user_id = $1 AND LOWER(name) = LOWER($2) LIMIT 1;",
            user_id, entity_name,
        )
        fact_ids = list(node["fact_ids"] or []) if node else []
        if fact_ids:
            rows = await conn.fetch(
                """
                SELECT id, fact_text, memory_type, belief_confidence, valid_from, valid_to,
                       created_at, is_active
                FROM atomic_facts
                WHERE user_id = $1 AND id = ANY($2)
                ORDER BY is_active DESC, belief_confidence DESC, created_at DESC
                LIMIT $3;
                """,
                user_id, fact_ids, limit,
            )
            if rows:
                # Prefer active facts; only fall back to inactive (deactivated-by-augmentation)
                # originals when no active fact links to this node.
                active = [dict(r) for r in rows if r["is_active"]]
                return active if active else [dict(r) for r in rows]
        # fallback: distinctive-token ILIKE (augmentation keeps core tokens)
        tokens = [t for t in entity_name.replace("-", " ").split() if len(t) > 2]
        if not tokens:
            tokens = [entity_name]
        conds = " OR ".join(f"LOWER(fact_text) LIKE '%' || LOWER(${i}) || '%'" for i in range(2, 2 + len(tokens)))
        params: List[Any] = [user_id, *tokens, limit]
        rows = await conn.fetch(
            f"""
            SELECT id, fact_text, memory_type, belief_confidence, valid_from, valid_to,
                   created_at, is_active
            FROM atomic_facts
            WHERE user_id = $1 AND is_active = TRUE AND ({conds})
            ORDER BY belief_confidence DESC, created_at DESC
            LIMIT ${len(params)};
            """,
            *params,
        )
        return [dict(r) for r in rows]


async def multihop_recall(
    user_id: str,
    query: str,
    seed_hits: Sequence[Any],
    *,
    max_facts: int = 8,
    max_hops: int = 3,
) -> List[Dict[str, Any]]:
    """Expand recall beyond the vector top-k using N-hop (default 3) entity-graph traversal.

    Entities are resolved with anaphora handling (H2) so "the company that builds KareOS"
    seeds the traversal correctly. Returns de-duplicated candidate fact dicts, ordered
    with the most chain-terminal fact first when it clearly answers the question (H3).
    Best-effort — never raises.
    """
    async with measure_latency("agi.multihop.multihop_recall"):
        try:
            from memory.graph.query_subgraph import query_subgraph
            from agi.anaphora import resolve_query_entities

            seed_ids = {getattr(h, "id", None) for h in seed_hits}
            seen: set = set()
            collected: List[Dict[str, Any]] = []
            depth: Dict[Any, int] = {}  # fact_id -> min hop at which discovered

            entities = await resolve_query_entities(query, user_id)
            if not entities:
                return collected

            # hop 1: entities directly mentioned / resolved in the query
            frontier = list(entities)
            for hop in range(1, max(1, max_hops) + 1):
                next_frontier: List[str] = []
                for ent in frontier:
                    sg = await query_subgraph(user_id, ent)
                    if sg is None:
                        for f in await _facts_for_entity(user_id, ent):
                            if f["id"] not in seen:
                                seen.add(f["id"]); collected.append(f)
                                depth.setdefault(f["id"], hop)
                        continue
                    for f in await _facts_for_entity(user_id, ent):
                        if f["id"] not in seen:
                            seen.add(f["id"]); collected.append(f)
                            depth.setdefault(f["id"], hop)
                    for edge in (sg.outgoing_edges or []) + (sg.incoming_edges or []):
                        neigh = edge.get("target_name") or edge.get("source_name")
                        if neigh and neigh.lower() not in {e.lower() for e in next_frontier}:
                            next_frontier.append(neigh)
                frontier = next_frontier
                if not frontier:
                    break

            # de-dup by id
            out: List[Dict[str, Any]] = []
            for f in collected:
                if f["id"] not in {x["id"] for x in out}:
                    out.append(f)
            out = out[:max_facts]
            if not out:
                return out
            # H3: promote the fact that best answers the relational question, judged by
            # the LLM (no hardcoded verb list) so it generalizes to new phrasings.
            return await rank_for_query(out, query, depth, known_entities=set(entities))
        except Exception as e:
            log_error(f"multihop_recall skipped: {e}")
            return []


async def rank_for_query(
    facts: List[Dict[str, Any]],
    query: str,
    depth: Dict[Any, int],
    known_entities: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Reorder candidate facts so the one that best answers `query` leads.

    Fused ranking via **Reciprocal Rank Fusion (RRF)** — the research-backed best
    (cognee ranks bm25+vector+summary with RRF; Graphiti uses cross-encoder+recency).
    RRF combines independent rank lists with ``Σ 1/(k+rank+1)`` and is IMMUNE to the
    score-scale problem that hand-weighted sums suffer: the ms-marco cross-encoder
    emits 0.998 vs 0.025 for unrelated-vs-answer facts, which no fixed weight separates
    cleanly. RRF only needs ORDINAL ranks, so scale mismatch is irrelevant. Signals:

      * ce_rank    — local cross-encoder relevance rank (Mem0/Graphiti reranker, on-device)
      * depth_rank — graph chain-position rank (deeper = chain terminus = the answer)
      * lex_rank   — query relational-token overlap rank
      * recency    — Graphiti-style: newer facts (valid_from) get a small additive boost

    Final RRF score is multiplied by an importance factor (cognee's
    ``0.75 + 0.5*belief_confidence``) so high-confidence facts lead ties. No keyword/
    verb vocabulary; generalizes to any phrasing. Falls back to lexical+depth RRF if
    the reranker is unavailable. Never raises.
    """
    if not facts:
        return facts

    q_tokens = {t.lower() for t in query.lower().split() if len(t) > 2}
    # query "relational" tokens = those that are not stopwords and not resolved entities
    from agi.anaphora import resolve_query_entities
    try:
        resolved = set(r.lower() for r in await resolve_query_entities(query, facts[0].get("user_id", "")) or [])
    except Exception:
        resolved = set()
    rel_tokens = {t for t in q_tokens if t not in resolved}

    max_depth = max((depth.get(f["id"], 1) for f in facts), default=1) or 1

    # --- signal 1: local cross-encoder relevance rank (one batched call if available)
    ce_scores: Dict[Any, float] = {}
    try:
        from memory.atomic.reranker import get_reranker
        reranker = get_reranker()
        if reranker is not None:
            docs = [{"fact_text": str(f.get("fact_text", ""))} for f in facts]
            scored = reranker.rerank(query, docs)
            for f, sdoc in zip(facts, scored):
                ce_scores[f["id"]] = float(sdoc.get("rerank_score", 0.0))
    except Exception as e:
        log_error(f"multihop reranker unavailable, lexical fallback: {e}")

    def _lex(f: Dict[str, Any]) -> float:
        ft = str(f.get("fact_text", "")).lower()
        if not rel_tokens:
            return 0.0
        return sum(1 for t in rel_tokens if t in ft) / len(rel_tokens)

    # --- asking-verb disambiguator: the question's main verb identifies the answer
    # relation. "who CO-FOUNDED X" -> the answer fact must contain "co-founded". This is
    # query-term matching (no keyword list) and is the principled primary signal for
    # relational questions, independent of fragile traversal-depth assignment.
    _SKIP = {"who", "what", "which", "the", "a", "an", "that", "this", "these",
             "does", "did", "do", "is", "are", "was", "were", "of", "in", "on",
             "to", "for", "with", "and", "or", "from", "at", "by", "company",
             "product", "person", "people", "thing", "things"}
    asking = None
    _known = {e.lower() for e in (known_entities or set())}
    for w in query.lower().split():
        if w in _SKIP or w in _known:
            continue
        if len(w) > 2:
            asking = w
            break

    def _has_asking(f: Dict[str, Any]) -> bool:
        if not asking:
            return False
        ft = str(f.get("fact_text", "")).lower()
        if asking in ft:
            return True
        # light lemmatization for verb forms (mentored->mentor, builds->build)
        for form in (asking.rstrip("ed"), asking.rstrip("s"), asking.rstrip("ing")):
            if len(form) > 2 and form in ft:
                return True
        return False

    # --- build ordinal rank lists for each signal (1 = best)
    def _rank_by(key_fn) -> Dict[Any, int]:
        ordered = sorted(facts, key=key_fn, reverse=True)
        return {f["id"]: i + 1 for i, f in enumerate(ordered)}

    asking_rank = _rank_by(lambda f: 1 if _has_asking(f) else 0)
    ce_rank = _rank_by(lambda f: ce_scores.get(f["id"], 0.0))
    depth_rank = _rank_by(lambda f: depth.get(f["id"], 1) / max_depth)
    lex_rank = _rank_by(_lex)

    # recency: newer valid_from -> higher (Graphiti-style temporal signal)
    def _recency(f: Dict[str, Any]) -> float:
        vf = f.get("valid_from")
        try:
            if vf:
                from datetime import datetime, timezone
                ref = vf if isinstance(vf, datetime) else datetime.fromisoformat(str(vf))
                return ref.timestamp()
        except Exception:
            pass
        return 0.0
    recency_rank = _rank_by(_recency)

    RRF_K = 60  # cognee uses 30..60; 60 is safe for small candidate sets
    _rrf_k = max(30, min(60, RRF_K))

    def _rrf(*ranks: int) -> float:
        return sum(1.0 / (_rrf_k + r + 1) for r in ranks if r > 0)

    # Primary order = RRF over PRINCIPLED signals, with the asking-verb as the lead
    # disambiguator for relational questions. The cross-encoder is a TIEBREAKER only:
    # with a small candidate set RRF scores are near-identical (~0.016 each), so any
    # additive ce term would let the (often wrong, MS-MARCO) cross-encoder dominate.
    # Graph structure + the question's own verb must lead; ce only refines ties.
    def _primary(f: Dict[str, Any]) -> float:
        return _rrf(asking_rank.get(f["id"], 0), depth_rank.get(f["id"], 0),
                    lex_rank.get(f["id"], 0), recency_rank.get(f["id"], 0))

    def _score(f: Dict[str, Any]) -> float:
        conf = f.get("belief_confidence")
        try:
            conf = float(conf) if conf is not None else 0.5
        except Exception:
            conf = 0.5
        importance = 0.75 + 0.5 * max(0.0, min(1.0, conf))
        s = _primary(f) * importance
        # Decisive asking-verb boost: when a fact contains the question's main verb
        # (co-founded / prefer / live ...), it MUST lead — this is the relational
        # disambiguator and must not be drowned out by the (often wrong, MS-MARCO)
        # cross-encoder under RRF's near-tie conditions (k=60 makes all RRF terms ~equal).
        if asking_rank.get(f["id"], 0) == 1:
            s += 1.0
        return s

    try:
        # sort by (primary*importance desc, cross-encoder score desc as tiebreaker)
        facts.sort(key=lambda f: (_score(f), ce_scores.get(f["id"], 0.0)), reverse=True)
    except Exception:
        pass
    return facts


def promote_answer(
    facts: List[Dict[str, Any]],
    query: str,
    depth: Dict[Any, int],
) -> List[Dict[str, Any]]:
    """Sync fallback retained for callers that can't await; delegates to lexical scoring."""
    q_tokens = {t.lower() for t in query.lower().split() if len(t) > 2}

    def score(f: Dict[str, Any]) -> float:
        ft = str(f.get("fact_text", "")).lower()
        overlap = len(q_tokens & set(ft.split())) / max(1.0, len(q_tokens))
        d = depth.get(f["id"], 1)
        return (d * 0.5) + overlap

    return sorted(facts, key=score, reverse=True)


__all__ = ["multihop_recall", "multihop_confidence", "promote_answer"]
