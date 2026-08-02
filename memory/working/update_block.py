"""Layer 2 (Letta): Dynamically update a specific working memory block."""

import json
from infrastructure.db import DatabasePool
from schemas import WorkingMemoryBlocks, WorkingMemoryRecord
from memory.working.get_blocks import get_blocks
from utils import measure_latency, log_working, log_error
from core.exceptions import StorageOperationError, MemoryBlockNotFoundError

async def update_block(user_id: str, block_name: str, content: str) -> WorkingMemoryRecord:
    """Update a specific named block in working memory."""
    async with measure_latency(f"memory.working.update_block ({block_name})"):
        record = await get_blocks(user_id)
        current_blocks = record.blocks.model_dump()
        
        if block_name not in current_blocks:
            raise MemoryBlockNotFoundError(f"Block '{block_name}' does not exist in working memory blocks.")

        current_blocks[block_name] = content
        updated_blocks = WorkingMemoryBlocks(**current_blocks)

        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO working_memory (user_id, blocks, updated_at)
                    VALUES ($1, $2::jsonb, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id)
                    DO UPDATE SET blocks = $2::jsonb, updated_at = CURRENT_TIMESTAMP
                    RETURNING user_id, blocks, updated_at;
                    """,
                    user_id,
                    updated_blocks.model_dump_json()
                )
                assert row is not None, "Working memory block update failed"
                log_working(f"Updated block [bold white]'{block_name}'[/bold white] for user [bold white]{user_id}[/bold white]")
                return WorkingMemoryRecord(
                    user_id=row["user_id"],
                    blocks=updated_blocks,
                    updated_at=row["updated_at"]
                )
        except Exception as e:
            log_error(f"Failed to update working memory block: {e}")
            raise StorageOperationError(f"Working memory update error: {e}")
