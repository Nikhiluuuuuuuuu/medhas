"""Layer 3 (Mem0): Full CRUD surface — get/update/delete/delete_all/history/reset.

Mirrors Mem0's Memory API: get(memory_id), update(memory_id, text), delete(memory_id),
delete_all(user_id, agent_id, run_id), history(memory_id), reset(). All operations are
soft (is_active flag) except history, which appends a versioned row.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID
from medhas.storage import DatabasePool
from medhas.memory.atomic.deactivate_fact import deactivate_fact
from medhas.memory.atomic.insert_fact import insert_fact
from medhas.schemas import AtomicFactSchema
from medhas.utils import measure_latency, log_atomic, log_error
from medhas.core.exceptions import StorageOperationError, MemoryBlockNotFoundError
from medhas.memory.atomic.json_utils import _coerce_json


async def get_memory(fact_id: UUID, user_id: Optional[str] = None) -> AtomicFactSchema:
    """Fetch a single active memory by id (Mem0 `get`)."""
    async with measure_latency("memory.atomic.get_memory"):
        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, user_id, session_id, agent_id, run_id, fact_text, categories,
                           memory_type, metadata, importance_score, is_active, created_at, expired_at
                    FROM atomic_facts WHERE id = $1 AND is_active = TRUE
                      AND ($2::text IS NULL OR user_id = $2);
                    """,
                    fact_id, user_id,
                )
                if not row:
                    raise MemoryBlockNotFoundError(f"Memory {fact_id} not found")
                return AtomicFactSchema(
                    id=row["id"], user_id=row["user_id"], session_id=row["session_id"],
                    agent_id=row["agent_id"], run_id=row["run_id"], fact_text=row["fact_text"],
                    categories=list(row["categories"] or []), memory_type=row["memory_type"],
                    metadata=_coerce_json(row["metadata"]), importance_score=float(row["importance_score"]),
                    is_active=row["is_active"], created_at=row["created_at"], expired_at=row["expired_at"],
                )
        except MemoryBlockNotFoundError:
            raise
        except Exception as e:
            log_error(f"get_memory failed: {e}")
            raise StorageOperationError(f"get_memory error: {e}")


async def update_memory(
    fact_id: UUID,
    new_text: str,
    user_id: Optional[str] = None,
    categories: Optional[List[str]] = None,
    memory_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AtomicFactSchema:
    """Update an existing memory (Mem0 `update`): soft-deactivate old, insert new with same scope."""
    async with measure_latency("memory.atomic.update_memory"):
        try:
            old = await get_memory(fact_id, user_id)
            # Soft-deactivate the prior version (preserves history).
            await deactivate_fact(fact_id)
            updated = await insert_fact(
                old.user_id, new_text,
                session_id=old.session_id, agent_id=old.agent_id, run_id=old.run_id,
                categories=categories if categories is not None else old.categories,
                memory_type=memory_type or old.memory_type,
                metadata=metadata if metadata is not None else old.metadata,
            )
            log_atomic(f"Updated memory {fact_id} -> {updated.id}")
            return updated
        except Exception as e:
            log_error(f"update_memory failed: {e}")
            raise StorageOperationError(f"update_memory error: {e}")


async def delete_memory(fact_id: UUID, user_id: Optional[str] = None) -> None:
    """Delete a single memory (Mem0 `delete`): soft-deactivate."""
    async with measure_latency("memory.atomic.delete_memory"):
        try:
            await get_memory(fact_id, user_id)  # existence check
            await deactivate_fact(fact_id)
            log_atomic(f"Deleted memory {fact_id}")
        except Exception as e:
            log_error(f"delete_memory failed: {e}")
            raise StorageOperationError(f"delete_memory error: {e}")


async def delete_all(
    user_id: str,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> int:
    """Delete all scoped memories (Mem0 `delete_all`): soft-deactivate matching active facts."""
    async with measure_latency("memory.atomic.delete_all"):
        try:
            async with DatabasePool.acquire() as conn:
                clause = "user_id = $1 AND is_active = TRUE"
                params: List[Any] = [user_id]
                i = 2
                for col, val in (("agent_id", agent_id), ("run_id", run_id)):
                    if val is not None:
                        params.append(val)
                        clause += f" AND {col} = ${i}"
                        i += 1
                status = await conn.execute(
                    f"UPDATE atomic_facts SET is_active = FALSE, expired_at = CURRENT_TIMESTAMP WHERE {clause};",
                    *params,
                )
                n = int(status.split()[-1]) if status else 0
                log_atomic(f"delete_all removed {n} memories for user {user_id}")
                return n
        except Exception as e:
            log_error(f"delete_all failed: {e}")
            raise StorageOperationError(f"delete_all error: {e}")


async def memory_history(fact_id: UUID) -> List[Dict[str, Any]]:
    """Return the version history of a memory (Mem0 `history`)."""
    async with measure_latency("memory.atomic.memory_history"):
        try:
            async with DatabasePool.acquire() as conn:
                # Current + any soft-deactivated prior versions sharing scope text lineage.
                rows = await conn.fetch(
                    """
                    SELECT id, fact_text, is_active, created_at, expired_at
                    FROM atomic_facts
                    WHERE id = $1 OR (user_id = (SELECT user_id FROM atomic_facts WHERE id=$1) AND fact_text = (SELECT fact_text FROM atomic_facts WHERE id=$1))
                    ORDER BY created_at ASC;
                    """,
                    fact_id,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            log_error(f"memory_history failed: {e}")
            return []


async def reset_user(user_id: str) -> None:
    """Reset all of a user's memory: deactivate facts + soft-close graph edges (Mem0 `reset`)."""
    async with measure_latency("memory.atomic.reset_user"):
        try:
            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    "UPDATE atomic_facts SET is_active = FALSE, expired_at = CURRENT_TIMESTAMP WHERE user_id = $1;",
                    user_id,
                )
                await conn.execute(
                    "UPDATE graph_edges SET valid_to = CURRENT_TIMESTAMP, expired_at = CURRENT_TIMESTAMP WHERE user_id = $1 AND valid_to IS NULL;",
                    user_id,
                )
                await conn.execute(
                    "UPDATE working_memory SET blocks = $2::jsonb WHERE user_id = $1;",
                    user_id, "{}",
                )
                log_atomic(f"Reset all memory for user {user_id}")
        except Exception as e:
            log_error(f"reset_user failed: {e}")
            raise StorageOperationError(f"reset_user error: {e}")
