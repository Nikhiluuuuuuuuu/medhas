"""Layer 3 (Mem0): Insert atomic fact with Mem0-style dedup & LLM decision matrix."""

import hashlib
import json
from typing import Optional, List, Dict, Any
from uuid import UUID
from infrastructure.db import DatabasePool
from infrastructure.llm import FastEmbeddingProvider, GroqLLMProvider
from memory.atomic.search_facts import search_facts
from memory.atomic.deactivate_fact import deactivate_fact
from memory.atomic.decision_matrix import decide_fact_action
from schemas import AtomicFactSchema
from config import settings
from utils import measure_latency, log_atomic, log_error
from core.exceptions import StorageOperationError
from memory.atomic.json_utils import _coerce_json
from memory.atomic.hashing import content_hash

embedder = FastEmbeddingProvider()


def _content_hash(text: str) -> str:
    return content_hash(text)


async def insert_fact(
    user_id: str,
    fact_text: str,
    is_reflection: bool = False,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    categories: Optional[List[str]] = None,
    memory_type: str = "semantic",
    metadata: Optional[Dict[str, Any]] = None,
) -> AtomicFactSchema:
    """Insert fact into atomic_facts with Mem0-style primary hash dedup + LLM decision matrix.

    Pipeline (mirrors Mem0's V3 batch pipeline, main.py:891-1177):
      1. MD5 hash dedup — if an ACTIVE fact with the same hash exists, treat as NO_CHANGE.
      2. Retrieve top semantically-similar candidates (vector + BM25).
      3. LLM decision matrix classifies ADD / UPDATE / DELETE / NO_CHANGE (with cosine guardrails).
      4. On UPDATE/DELETE, soft-deactivate the prior fact; insert the new fact.
    """
    fact_text = fact_text.strip()
    if not fact_text:
        raise StorageOperationError("Cannot insert empty fact")

    async with measure_latency("memory.atomic.insert_fact"):
        try:
            content_hash = _content_hash(fact_text)

            async with DatabasePool.acquire() as conn:
                # 1. PRIMARY DEDUP: exact content hash (Mem0 main.py:991 md5 dedup)
                if settings.FACT_HASH_DEDUP:
                    existing_hash = await conn.fetchrow(
                        """
                        SELECT id, fact_text, session_id, agent_id, created_at
                        FROM atomic_facts
                        WHERE user_id = $1 AND is_active = TRUE AND content_hash = $2
                        LIMIT 1;
                        """,
                        user_id, content_hash,
                    )
                    if existing_hash:
                        log_atomic(f"Mem0 hash dedup: NO_CHANGE for fact: '{fact_text}'")
                        return AtomicFactSchema(
                            id=existing_hash["id"],
                            user_id=user_id,
                            session_id=existing_hash["session_id"],
                            agent_id=existing_hash["agent_id"],
                            fact_text=existing_hash["fact_text"],
                            is_active=True,
                            created_at=existing_hash["created_at"],
                        )

                # 2. Retrieve candidates for the decision matrix
                candidates = await search_facts(
                    user_id, fact_text, limit=5,
                    similarity_threshold=settings.FACT_SEMANTIC_UPDATE_THRESHOLD - 0.1,
                    session_id=session_id, agent_id=agent_id,
                )

                # 3. LLM decision matrix (falls back to ADD if LLM unavailable)
                decision = await decide_fact_action(fact_text, candidates, llm=GroqLLMProvider())

                action = decision.get("action", "ADD")
                target_id = decision.get("target_id")

                # Cosine guardrails: if a candidate is near-identical, force NO_CHANGE/UPDATE
                near_dup = next(
                    (c for c in candidates if float(c.similarity) >= settings.FACT_SEMANTIC_DUP_THRESHOLD),
                    None,
                )
                if near_dup and action == "ADD":
                    action = "NO_CHANGE"
                    target_id = near_dup.id

                if action in ("NO_CHANGE",):
                    log_atomic(f"Mem0 Matrix: NO_CHANGE for fact: '{fact_text}'")
                    target = next((c for c in candidates if c.id == target_id), candidates[0]) if candidates else None
                    if target is None:
                        # Hash miss but cosine near-dup with no retrievable target: still insert to be safe.
                        action = "ADD"
                    else:
                        return AtomicFactSchema(
                            id=target.id, user_id=user_id,
                            session_id=target.session_id, agent_id=target.agent_id,
                            fact_text=target.fact_text, is_active=True, created_at=target.created_at,
                        )

                if action in ("UPDATE", "DELETE"):
                    if target_id:
                        # Subject-mismatch guard (Graphiti-style): never deactivate a target
                        # whose subject entity differs from the incoming fact's subject. The
                        # decision matrix can over-merge cross-subject neighbours that merely
                        # share tokens (e.g. "priya launched lumina" vs "priya mentors rahul…
                        # lumina"); deactivating the original would lose retrievable memory.
                        # Keep both facts and resolve via valid_from instead of deletion.
                        target_cand = next((c for c in candidates if c.id == target_id), None)
                        if target_cand is not None:
                            from agi.entities import query_entities
                            new_subj = set(s.lower() for s in query_entities(fact_text))
                            tgt_subj = set(s.lower() for s in query_entities(getattr(target_cand, "fact_text", "")))
                            shared = new_subj & tgt_subj
                            if not shared:
                                log_atomic(
                                    f"Mem0 Matrix: {action} on different-subject target "
                                    f"{target_id}; keeping both (ADD)"
                                )
                                action = "ADD"
                                target_id = None
                        if action in ("UPDATE", "DELETE") and target_id is not None:
                            log_atomic(f"Mem0 Matrix: Deactivating conflicting fact {target_id} ({action})")
                            await deactivate_fact(target_id)
                    else:
                        # LLM asked to UPDATE/DELETE but named no resolvable target — do NOT
                        # silently insert a duplicate. Require a valid target; otherwise treat as
                        # ADD only if no near-duplicate candidate exists, else NO_CHANGE.
                        log_atomic(
                            f"Mem0 Matrix: {action} requested with no target_id; "
                            f"falling back to NO_CHANGE/ADD guardrail"
                        )
                        if near_dup:
                            action = "NO_CHANGE"
                            target_id = near_dup.id
                        else:
                            action = "ADD"

            # 4. Embed + store new fact
            embedding = await embedder.embed_text(fact_text)
            vector_str = f"[{','.join(str(x) for x in embedding)}]"

            # H1: derive valid_from from a date in the fact text when present (open LLM
            # extraction, regex fallback), so temporal recall ('before T') is grounded
            # without manual backdating and without a closed date-format vocabulary.
            from agi.llm_extract import extract_date_open
            extracted_dt = await extract_date_open(fact_text)

            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO atomic_facts (user_id, session_id, agent_id, run_id, fact_text, embedding, is_active, content_hash, categories, memory_type, metadata, valid_from)
                    VALUES ($1, $2, $3, $4, $5, $6::vector, TRUE, $7, $8, $9, $10::jsonb, COALESCE($11, CURRENT_TIMESTAMP))
                    RETURNING id, user_id, session_id, agent_id, run_id, fact_text, categories, memory_type, metadata, is_active, created_at;
                    """,
                    user_id, session_id, agent_id, run_id, fact_text, vector_str, content_hash,
                    categories or [], memory_type, json.dumps(metadata or {}),
                    extracted_dt,
                )
                assert row is not None, "Failed to insert atomic fact"
                fact = AtomicFactSchema(
                    id=row["id"], user_id=row["user_id"], session_id=row["session_id"],
                    agent_id=row["agent_id"], run_id=row["run_id"], fact_text=row["fact_text"],
                    categories=list(row["categories"] or []), memory_type=row["memory_type"],
                    metadata=_coerce_json(row["metadata"]),
                    is_active=row["is_active"], created_at=row["created_at"]
                )
                log_atomic(f"Inserted new active fact: '{fact_text}'")
                return fact
        except Exception as e:
            log_error(f"Failed to insert fact: {e}")
            # G1 fix: never silently return None (callers assert fact.id is not None and
            # would otherwise lose the fact with no error). Surface the failure.
            from core.exceptions import StorageOperationError
            if isinstance(e, StorageOperationError):
                raise
            raise StorageOperationError(f"insert_fact failed: {e}") from e
