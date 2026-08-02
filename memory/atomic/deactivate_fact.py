"""Layer 3 (Mem0): Soft-delete an outdated or contradicted atomic fact."""

from uuid import UUID
from infrastructure.db import DatabasePool
from utils import measure_latency, log_atomic, log_error
from core.exceptions import StorageOperationError

async def deactivate_fact(fact_id: UUID) -> bool:
    """Mark an existing fact as inactive (soft-delete)."""
    async with measure_latency("memory.atomic.deactivate_fact"):
        try:
            async with DatabasePool.acquire() as conn:
                status = await conn.execute(
                    """
                    UPDATE atomic_facts
                    SET is_active = FALSE, expired_at = CURRENT_TIMESTAMP
                    WHERE id = $1 AND is_active = TRUE;
                    """,
                    fact_id
                )
                log_atomic(f"Deactivated outdated fact [bold white]{fact_id}[/bold white]")
                return "UPDATE 1" in status
        except Exception as e:
            log_error(f"Failed to deactivate fact: {e}")
            raise StorageOperationError(f"Deactivate fact error: {e}")
