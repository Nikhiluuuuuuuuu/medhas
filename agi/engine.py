"""Medhas MemoryEngine — the unified AGI-memory orchestrator (roadmap 0–37).

This is a thin, additive orchestration layer on top of the existing store modules.
It does NOT replace memory/ or pipeline/; it composes them so every roadmap
enhancement is reachable from one entry point, and exposes a single `remember`
and `recall` surface that bakes in admission, memory-type routing, contradiction
handling, reconsolidation, security and metamemory.

Key invariant (user's non-destructive rule): existing modules are only ever *called*,
never rewritten. Every new behaviour lives in agi/*.py.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID
import asyncio
import re

# roadmap memory-type taxonomy + routing
from agi.memory_types import MemoryType, route, is_valid

# admission (E32)
from agi.admission import evaluate_admission

# bitemporal / belief / provenance (E10-E12)
from agi.bitemporal import (
    revise_fact_belief, invalidate_fact, mark_contradiction,
    facts_valid_at, fact_provenance, bayesian_update,
)
# G3 temporal recall (before/after intent)
from agi.temporal import recall_before_after

# forgetting / salience / reconsolidation / spaced review (E9, E13-E15, E28, E33)
from agi.forgetting import (
    run_forgetting_sweep, reconsolidate, set_affect, protect_core_memories,
    is_protected, schedule_review, due_for_review, retention,
)

# security (E34)
from agi.security import (
    sign_write, verify_write, trust_for_source, check_poisoning, quarantine_fact,
    release_quarantine, list_quarantined, forget, sandbox_for_tools,
)

# consolidation (E2, E3, E7, E8)
from agi.consolidation import (
    compress_episodes, induce_skills, evolve_memory_network, run_consolidation,
)

# scheduler (E7 runtime) + rehearsal (E36)
from agi.scheduler import scheduler, rehearse

# interference / WM eviction / recognition (E5, E35, E37)
from agi.interference import (
    interference_matrix, resolve_interference, evict_working_memory, recognize, eviction_scores,
)

# metacognitive routing + abstention + self-improvement (E16-E18)
from agi.metacognitive import (
    route_query, should_abstain, RetrievalOutcome, critic, log_memory_event,
)

# user model + temporal/causal API (E19, E20)
from agi.usermodel import (
    build_user_model, get_user_model, timeline, what_changed, why_chain,
)

# prospective / sensory / metamemory (E27, E29, E31)
from agi.prospective import (
    add_intention, check_cues, complete_intention, list_intentions,
)
from agi.metamemory import assess as metamemory_assess, known_unknowns, knowledge_map
from agi.cognition.embodiment import BodyModel  # cognition subsystem (forward-ref in think())
from agi.sensory import buffer_percept, promote_percepts, sweep_expired, list_buffer, attention_filter

# ingest / chunking / extraction (E4)
from agi.ingest import chunk_text, extract_facts, extract_from_document, detect_duplicates

# auth / rate-limit (E22, E23)
from agi.auth import authenticate, authorize, rate_limiter

# scaling / export (E25, E26)
from agi.scaling import ensure_hot_view, refresh_hot_view, partition_report
from agi.export import export_user_memory, export_to_file, import_user_memory

# eval (E21)
from agi.eval import run_eval_suite, temporal_consistency_check, EvalCase

from infrastructure.db import DatabasePool
from utils import log_atomic, log_error, measure_latency


class MemoryEngine:
    """Unified AGI-memory facade. Compose, don't replace."""

    def __init__(self) -> None:
        self.version = "agi-roadmap-37"

    # ============================================================ REMEMBER (write)

    async def remember_batch(
        self,
        user_id: str,
        facts: List[str],
        *,
        delay: float = 1.2,
        **kw,
    ) -> List[Dict[str, Any]]:
        """Throttled batch write.

        Groq enforces low RPM; remembering many facts back-to-back trips 429 rate
        limits (the provider retries with backoff, but a large batch can still exceed
        the test/CLI timeout). This writes facts serially with a short inter-call delay
        so a full corpus ingests cleanly. Best-effort: collects each result, never raises
        on an individual failure.
        """
        results: List[Dict[str, Any]] = []
        for i, f in enumerate(facts):
            try:
                results.append(await self.remember(user_id, f, **kw))
            except Exception as e:  # keep going on a single bad fact
                log_error(f"remember_batch item {i} failed: {e}")
                results.append({"error": str(e)})
            if delay and i < len(facts) - 1:
                await asyncio.sleep(delay)
        return results

    async def remember(
        self,
        user_id: str,
        text: str,
        *,
        session_id: Optional[UUID] = None,
        memory_type: str = "semantic",
        source: str = "user",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        affect: Optional[Dict[str, float]] = None,
        force_admit: bool = False,
    ) -> Dict[str, Any]:
        """Write a memory with the full roadmap pipeline applied (E1,E10-E12,E32,E34)."""
        async with measure_latency("engine.remember"):
            if not is_valid(memory_type):
                memory_type = "semantic"

            # --- E34 integrity signature
            signature = sign_write(user_id, text)

            # --- E32 admission control
            from memory.atomic import search_facts
            candidates = await search_facts(user_id, text, limit=5)
            source_trust = trust_for_source(source)
            decision = evaluate_admission(
                text, candidates, source_trust=source_trust,
                provenance_kind="implicit_inferred" if source == "inferred" else "explicit",
                force=force_admit,
            )
            if decision.action == "DROP" and not force_admit:
                await log_memory_event(user_id, "admission_drop", {
                    "score": decision.score, "text": text[:80]})
                return {"status": "dropped", "reason": decision.reason, "score": decision.score}

            # --- E34 poisoning / contradiction check
            poison = await check_poisoning(user_id, text, candidates, source_trust=source_trust)

            from memory.atomic import insert_fact
            fact = await insert_fact(
                user_id, text,
                session_id=session_id, agent_id=agent_id,
                memory_type=memory_type,
                metadata={**(metadata or {}), "source": source},
            )
            assert fact.id is not None, "insert_fact returned no id"
            fact_id: UUID = fact.id

            # attach new AGI columns
            await self._attach_agi_meta(
                fact_id, signature, source, source_trust,
                provenance_kind="implicit_inferred" if source == "inferred" else "explicit",
            )
            if affect:
                await set_affect(fact_id, float(affect.get("valence", 0.0)),
                                 float(affect.get("arousal", 0.0)))

            # --- G2 build entity graph from the fact (non-fatal, best-effort)
            try:
                from agi.graph_build import build_fact_graph
                await build_fact_graph(
                    user_id, fact_id, text,
                    session_id=session_id, agent_id=agent_id,
                )
            except Exception as e:
                log_error(f"graph build skipped: {e}")

            # --- E10/E12 contradiction handling
            for c in candidates:
                ct = str(getattr(c, "fact_text", "")).lower()
                if ct and ct != text.lower() and await self._contradicts_semantic(text, ct):
                    await mark_contradiction(fact_id, c.id)
                    # lower the older fact's belief (Bayesian contradiction)
                    await revise_fact_belief(c.id, likelihood=0.6, supports=False)
                    # supersede the older fact: close its valid-time window AND soft-deactivate
                    # so it is excluded from "what is true now" while staying queryable
                    # historically (Zep/Graphiti temporal-invalidation pattern — coexist, don't delete).
                    try:
                        from infrastructure.db import DatabasePool
                        async with DatabasePool.acquire() as conn:
                            await conn.execute(
                                "UPDATE atomic_facts SET is_active = FALSE WHERE id = $1;",
                                c.id,
                            )
                    except Exception as deact_err:
                        log_error(f"contradiction deactivate skipped: {deact_err}")
                    try:
                        await invalidate_fact(c.id, invalidated_by=fact_id)
                    except Exception as e:
                        log_error(f"contradiction invalidate skipped: {e}")

            # --- E8 A-MEM evolution (retro-link + revise neighbours)
            try:
                await evolve_memory_network(user_id, fact_id, text, neighbours=candidates)
            except Exception as e:
                log_error(f"network evolution skipped: {e}")

            # --- E34 quarantine if flagged
            if poison.get("quarantine"):
                await quarantine_fact(fact_id, poison.get("reason", ""))
                await log_memory_event(user_id, "quarantine", {"id": str(fact_id)})
                return {"status": "quarantined", "id": str(fact_id), "reason": poison["reason"]}

            await log_memory_event(user_id, "remember", {
                "id": str(fact_id), "type": memory_type, "admission": decision.score})
            return {
                "status": "stored", "id": str(fact_id),
                "memory_type": memory_type, "admission_score": decision.score,
                "linked": len(candidates),
            }

    async def _attach_agi_meta(
        self, fact_id: UUID, signature: str, source: str,
        source_trust: float, provenance_kind: str,
    ) -> None:
        try:
            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE atomic_facts
                    SET write_signature = $2, source_trust = $3, provenance_kind = $4
                    WHERE id = $1;
                    """,
                    fact_id, signature, source_trust, provenance_kind,
                )
        except Exception as e:
            log_error(f"_attach_agi_meta failed: {e}")

    @staticmethod
    def _contradicts(a: str, b: str) -> bool:
        # Lightweight lexical opposition signal (e.g. "is" vs "is not").
        neg_patterns = (("is", "is not"), ("has", "has no"), ("always", "never"),
                        ("can", "cannot"), ("do", "do not"), ("will", "will not"))
        al, bl = a.lower(), b.lower()
        for p, n in neg_patterns:
            if (p in al and n in bl) or (n in al and p in bl):
                return True
        return False

    @staticmethod
    async def _contradicts_semantic(a: str, b: str) -> bool:
        """Semantic contradiction detection (offline, local embeddings).

        Two facts contradict when they assert different things about the SAME
        subject+relation, e.g. "Nikhil lives in Bangalore" vs "Nikhil lives in
        Hyderabad" (both are LIVES_IN, different objects) — these are mutually
        exclusive regardless of how lexically similar the sentences are. This mirrors
        human contradiction detection ("he can't live in both cities") and works
        WITHOUT an LLM call: it uses local FastEmbed embeddings + a structural check
        on shared relation verbs.

        Returns True when:
          (1) the two facts share a salient subject entity AND a relation verb but the
              objects differ (structural contradiction), OR
          (2) lexical negation (is/is not) is present, OR
          (3) same subject but embedding cosine < 0.45 (semantically divergent).
        """
        # Fast lexical negation gate.
        if MemoryEngine._contradicts(a, b):
            return True
        # Structural contradiction: same relation verb, different objects, shared subject.
        al, bl = a.lower(), b.lower()
        for verb in ("lives in", "lived in", "live in", "works at", "works for",
                     "prefers", "is", "was", "are", "were", "joined", "mentors",
                     "located in", "based in", "founded", "launched", "headquarters"):
            if verb in al and verb in bl:
                sub_a = al.split(verb)[0].strip()
                sub_b = bl.split(verb)[0].strip()
                obj_a = al.split(verb)[1].strip()
                obj_b = bl.split(verb)[1].strip()
                # shared subject (entity), different object -> contradiction
                if sub_a and sub_a == sub_b and obj_a and obj_b and obj_a != obj_b:
                    return True
        # Embedding-based divergence on a shared subject (catch-all).
        try:
            from agi.entities import query_entities
            ea, eb = set(e.lower() for e in query_entities(a)), set(e.lower() for e in query_entities(b))
            shared = ea & eb
            if not shared:
                return False
            from infrastructure.llm import FastEmbeddingProvider
            emb = FastEmbeddingProvider()
            va, vb = await emb.embed_text(a), await emb.embed_text(b)
            sim = sum(x * y for x, y in zip(va, vb)) / (
                (sum(x * x for x in va) ** 0.5) * (sum(y * y for y in vb) ** 0.5) + 1e-9
            )
            return sim < 0.45
        except Exception:
            return False

    # ============================================================= RECALL (read)

    async def recall(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        valid_at: Optional[datetime] = None,
        use_tools: bool = False,
        enforce_abstention: bool = True,
    ) -> Dict[str, Any]:
        """Retrieve memories with routing, reconsolidation, and calibrated abstention."""
        async with measure_latency("engine.recall"):
            # --- E5 recognition first (cheap content-hash gate)
            rec = await recognize(user_id, query)
            # --- E16 routing
            routing = route_query(query, recognized=rec.get("known", False))
            from memory.atomic import search_facts
            temporal: Optional[List[Dict[str, Any]]] = None
            extra: List[Dict[str, Any]] = []
            resolved: List[str] = []
            raw_ents: List[str] = []
            # --- E10 temporal scope when requested
            if valid_at is not None:
                facts = await facts_valid_at(user_id, valid_at, limit=limit)
                hits: Sequence[Any] = [type("Row", (), dict(f))() for f in facts]
            else:
                # --- G3: natural-language before/after intent (e.g. "before Hyderabad").
                # Detected here (not only via routing) so temporal answers lead over multi-hop.
                temporal_intent = any(w in query.lower() for w in (" before ", " after ", " prior to ", " since ", " when "))
                temporal: Optional[List[Dict[str, Any]]] = None
                extra: List[Dict[str, Any]] = []
                if temporal_intent:
                    temporal = await recall_before_after(user_id, query, limit=routing.top_k)
                if temporal:
                    hits = [type("Row", (), dict(f))() for f in temporal]
                else:
                    hits = await search_facts(user_id, query, limit=routing.top_k)
                    # --- G2/H2/H3: expand via N-hop entity graph when vector recall is
                    # thin OR the query resolves to more entities than its surface tokens
                    # (anaphora fired: "he", "the company that builds X", etc.). No closed
                    # keyword vocabulary — the resolution step decides what's relational.
                    # Always attempt; best-effort.
                    from agi.multihop import multihop_recall
                    from agi.anaphora import resolve_query_entities
                    from agi.entities import query_entities
                    # initialize so a 429-raised resolution still leaves these bound
                    # (otherwise the unconditional promotion below would NameError and skip)
                    resolved: List[str] = []
                    raw_ents: List[str] = []
                    relational = False
                    try:
                        resolved = await resolve_query_entities(query, user_id)
                        raw_ents = query_entities(query)
                        relational = (
                            len(hits) < 3
                            or len(resolved) > len(raw_ents)
                        )
                        # 429-proof fallback: when LLM resolution returned nothing but the
                        # query uses a pronoun (he/she/they), resolve to the most salient
                        # person via the graph (DB-only, no LLM). This keeps anaphoric
                        # promotion working even when the Groq key is rate-limited.
                        if not resolved and re.search(r"\b(he|she|they|his|her|their)\b", query.lower()):
                            try:
                                from agi.anaphora import _most_salient_person
                                p = await _most_salient_person(user_id)
                                if p:
                                    resolved = [p]
                                    relational = True
                            except Exception:
                                pass
                        if relational:
                            extra = await multihop_recall(
                                user_id, query, hits, max_facts=routing.top_k
                            )
                            if extra:
                                extra_objs = [type("Row", (), dict(f))() for f in extra]
                                # For relational/anaphoric questions the multi-hop answer is the
                                # desired one, so lead with the (already promoted) graph results.
                                merged = extra_objs + list(hits)
                                # de-dup by id
                                seen_ids = set()
                                hits = []
                                for h in merged:
                                    hid = getattr(h, "id", None)
                                    if hid not in seen_ids:
                                        seen_ids.add(hid); hits.append(h)
                            else:
                                # multihop returned nothing (e.g. entity resolution hit a 429 or
                                # the graph had no extra edges). Still promote the asking-verb fact
                                # within the vector results so relational/anaphoric answers lead
                                # even when traversal is unavailable. Convert Row->dict, promote,
                                # convert back.
                                try:
                                    from agi.multihop import rank_for_query
                                    hit_dicts = [_result_dict(h) for h in hits]
                                    promoted = await rank_for_query(
                                        hit_dicts, query,
                                        {getattr(h, "id", None): 1 for h in hits},
                                        known_entities=set(resolved),
                                    )
                                    hits = [type("Row", (), dict(f))() for f in promoted]
                                except Exception as e:
                                    log_error(f"asking-verb promote skipped: {e}")
                    except Exception as e:
                        log_error(f"multihop expand skipped: {e}")

                    # --- unconditional asking-verb promotion (H2/H3 robustness) ---
                    # Promote the fact containing the question's asking-verb (co-founded,
                    # prefer, live, ...) to rank 1 within the merged hits. The asking-verb is
                    # derived from the QUERY (not from resolved entities), so promotion runs
                    # even when entity resolution was blocked by a 429. known_entities (when
                    # available) only prevents mistaking an entity NAME for the verb.
                    try:
                        from agi.multihop import rank_for_query
                        _rd = [_result_dict(h) for h in hits]
                        promoted = await rank_for_query(
                            _rd, query,
                            {getattr(h, "id", None): 1 for h in hits},
                            known_entities=set(resolved),
                        )
                        hits = [type("Row", (), dict(f))() for f in promoted]
                    except Exception as e:
                        log_error(f"asking-verb promote skipped: {e}")

            # --- E15 reconsolidation on recall
            ids: List[UUID] = [getattr(h, "id") for h in hits if getattr(h, "id", None)]
            if ids:
                await reconsolidate(ids)

            # --- E34 sandbox for tool context
            if use_tools:
                hits = sandbox_for_tools(hits)

            # --- E18/E29 calibrated abstention
            # When relational/anaphoric/temporal EXPANSION succeeded we already have
            # evidence by a mechanism other than lexical overlap, so do not abstain
            # even if the lexical FoK signal is low (relational/temporal queries are
            # inherently low-overlap with their answer fact). Anaphora resolution
            # (e.g. "he" -> Nikhil) also counts as expanded evidence.
            anaphora_resolved = len(resolved) > len(raw_ents)
            expanded = bool(temporal) or (extra is not None and len(extra) > 0) or anaphora_resolved
            mm = await metamemory_assess(user_id, query, hits)
            if enforce_abstention and not expanded:
                should = should_abstain(hits, mm)
            else:
                should = {"abstain": False, "confidence": mm.get("fok", 1.0)}
            if should["abstain"]:
                await log_memory_event(user_id, "abstain", {"query": query[:80]})
                return {
                    "status": "abstained", "confidence": should["confidence"],
                    "reason": should.get("message", ""), "results": [],
                    "metamemory": mm,
                }
            return {
                "status": "ok", "strategy": routing.strategy,
                "confidence": mm.get("fok", 0.0),
                "results": [_result_dict(h) for h in hits],
                "metamemory": mm,
            }


    async def think(self, text: str, user_id: str, *, modality: str = "text",
                    body: Optional["BodyModel"] = None, action: Optional[str] = None,
                    action_params: Optional[dict] = None) -> Dict[str, Any]:
        """Cognition entry point: run the perception->reasoning->generalization->embodiment
        pipeline over one input. Returns the structured CognitiveResult plus ingestion hints.

        This is the thin loop that turns the memory engine into a cognitive agent. Offline-safe.
        """
        from agi.cognition import cognitive_step
        result = await cognitive_step(
            text, user_id, modality=modality, body=body,
            action=action, action_params=action_params,
        )
        # Optionally persist the perceived fact so downstream recall sees it.
        try:
            fact_text = result.percept.to_fact_text() if result.percept else text
            if fact_text:
                await self.remember(user_id, fact_text, source="cognitive")
        except Exception as e:
            result.notes.append(f"think ingest skipped: {e}")
        derived = [{"subject": s, "relationship": r, "object": o} for s, r, o in result.derived_facts]
        return {
            "percept": {
                "modality": result.percept.modality,
                "salience": result.percept.salience,
                "scene_type": result.percept.scene_type,
                "entities": result.percept.entities,
                "relations": [list(t) for t in result.percept.relations],
            } if result.percept else None,
            "derived_facts": derived,
            "schema_predictions": result.schema_predictions,
            "analogies": [list(t) for t in result.analogies],
            "action": result.action,
            "notes": result.notes,
        }

    # --------------------------------------------------------- meta operations

    async def consolidate(self, user_id: str) -> Dict[str, Any]:
        return await run_consolidation(user_id)

    async def forget_user(self, user_id: str, scope: Optional[str] = None, *, hard: bool = True) -> Dict[str, Any]:
        return await forget(user_id, scope, hard=hard)

    async def backup(self, user_id: str, path: Optional[str] = None) -> Dict[str, Any]:
        bundle = await export_user_memory(user_id)
        if path:
            export_to_file(bundle, path)
        return {"path": path, "rows": sum(len(v) for v in bundle.get("tables", {}).values())}

    async def build_profile(self, user_id: str) -> Dict[str, Any]:
        return await build_user_model(user_id)

    async def plan_intention(self, user_id: str, intent: str, **kw) -> UUID:
        return await add_intention(user_id, intent, **kw)

    async def fire_intentions(self, user_id: str, context: str = "") -> List[Dict[str, Any]]:
        return await check_cues(user_id, context)

def _result_dict(h) -> dict:
    """Serialize a recall hit to a plain dict, tolerant of the lightweight
    type('Row', (), {...})() wrapper used for temporal/multihop results (whose
    data lives in the class namespace, not the instance __dict__)."""
    d = getattr(h, "__dict__", None)
    if isinstance(d, dict) and d:
        return dict(d)
    out = {}
    for k in ("id", "fact_text", "memory_type", "belief_confidence",
              "valid_from", "valid_to", "created_at", "similarity", "rrf_score",
              "invalidated_by", "provenance_kind", "source_episode_id",
              "contradicted_by"):
        if hasattr(h, k):
            out[k] = getattr(h, k)
    if "fact_text" not in out and hasattr(h, "fact_text"):
        out["fact_text"] = h.fact_text
    return out or {"fact_text": getattr(h, "fact_text", "")}


#: shared engine instance
engine = MemoryEngine()
