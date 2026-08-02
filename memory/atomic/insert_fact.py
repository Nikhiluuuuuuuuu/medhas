"""Layer 3 (Mem0): Insert atomic fact with strict deduplication & conflict resolution."""

from typing import Optional
from uuid import UUID
from infrastructure.db import DatabasePool
from infrastructure.llm import FastEmbeddingProvider
from memory.atomic.search_facts import search_facts
from memory.atomic.deactivate_fact import deactivate_fact
from schemas import AtomicFactSchema
from utils import measure_latency, log_atomic, log_error
from core.exceptions import StorageOperationError

embedder = FastEmbeddingProvider()

async def evaluate_memory_decision_matrix(
    incoming_fact: str,
    existing_facts: list
) -> dict:
    """Mem0 LLM Memory Decision Matrix: Classifies incoming facts into ADD, UPDATE, DELETE, or NO_CHANGE actions."""
    if not existing_facts:
        return {"action": "ADD"}

    for old_fact in existing_facts:
        if old_fact.fact_text.lower().strip() == incoming_fact.lower().strip() or old_fact.similarity >= 0.90:
            return {"action": "NO_CHANGE", "target_id": old_fact.id}
        
        if old_fact.similarity > 0.75 and not incoming_fact.startswith("[Reflection]"):
            return {"action": "UPDATE", "target_id": old_fact.id}

    return {"action": "ADD"}

async def insert_fact(
    user_id: str,
    fact_text: str,
    is_reflection: bool = False,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None
) -> AtomicFactSchema:
    """Insert fact into atomic_facts with Mem0 decision matrix and ground-truth preservation across multi-scopes."""
    async with measure_latency("memory.atomic.insert_fact"):
        try:
            # 1. Deduplication & decision matrix check
            existing_similar = await search_facts(user_id, fact_text, limit=3, similarity_threshold=0.65, session_id=session_id, agent_id=agent_id)
            decision = await evaluate_memory_decision_matrix(fact_text, existing_similar)

            if decision["action"] == "NO_CHANGE":
                log_atomic(f"Mem0 Matrix: NO_CHANGE for fact: [bold white]'{fact_text}'[/bold white]")
                target = next((f for f in existing_similar if f.id == decision.get("target_id")), existing_similar[0])
                return AtomicFactSchema(
                    id=target.id,
                    user_id=user_id,
                    session_id=target.session_id,
                    agent_id=target.agent_id,
                    fact_text=target.fact_text,
                    is_active=True,
                    created_at=target.created_at
                )

            elif decision["action"] == "UPDATE" or decision["action"] == "DELETE":
                target_id = decision.get("target_id")
                if target_id:
                    log_atomic(f"Mem0 Matrix: Deactivating conflicting fact {target_id}")
                    await deactivate_fact(target_id)

            # 2. Embed new fact
            embedding = await embedder.embed_text(fact_text)
            vector_str = f"[{','.join(str(x) for x in embedding)}]"

            # 3. Store in Postgres
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO atomic_facts (user_id, session_id, agent_id, fact_text, embedding, is_active)
                    VALUES ($1, $2, $3, $4, $5::vector, TRUE)
                    RETURNING id, user_id, session_id, agent_id, fact_text, is_active, created_at;
                    """,
                    user_id,
                    session_id,
                    agent_id,
                    fact_text,
                    vector_str
                )
                assert row is not None, "Failed to insert atomic fact"
                fact = AtomicFactSchema(
                    id=row["id"],
                    user_id=row["user_id"],
                    session_id=row["session_id"],
                    agent_id=row["agent_id"],
                    fact_text=row["fact_text"],
                    is_active=row["is_active"],
                    created_at=row["created_at"]
                )
                log_atomic(f"Inserted new active fact: [bold white]'{fact_text}'[/bold white]")
                return fact
        except Exception as e:
            log_error(f"Failed to insert fact: {e}")
            raise StorageOperationError(f"Insert fact error: {e}")
