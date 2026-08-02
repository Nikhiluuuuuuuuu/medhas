"""Layer 2 (Letta): Fetch working memory JSONB prompt blocks."""

import json
from infrastructure.db import DatabasePool
from schemas import WorkingMemoryBlocks, MemoryBlock, WorkingMemoryRecord
from utils import measure_latency, logger
from core.exceptions import StorageOperationError

async def get_blocks(user_id: str) -> WorkingMemoryRecord:
    """Retrieve working memory blocks for a user, initializing defaults if absent."""
    async with measure_latency("memory.working.get_blocks"):
        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT user_id, blocks, updated_at
                    FROM working_memory
                    WHERE user_id = $1;
                    """,
                    user_id
                )
                if not row:
                    # Create default blocks
                    default_blocks = WorkingMemoryBlocks()
                    blocks_json = default_blocks.model_dump_json()
                    row = await conn.fetchrow(
                        """
                        INSERT INTO working_memory (user_id, blocks)
                        VALUES ($1, $2::jsonb)
                        RETURNING user_id, blocks, updated_at;
                        """,
                        user_id,
                        blocks_json
                    )
                
                assert row is not None, "Working memory row retrieval failed"
                raw_blocks = row["blocks"]
                if isinstance(raw_blocks, str):
                    block_dict = json.loads(raw_blocks)
                else:
                    block_dict = dict(raw_blocks)

                # Source of truth is the per-block "extra" namespace (full MemoryBlock
                # dicts). Rebuild the registry from it so custom blocks survive round-trips.
                block_map = {}
                for label, val in block_dict.items():
                    if isinstance(val, dict) and "value" in val and "label" in val:
                        block_map[label] = MemoryBlock(**val)
                    elif isinstance(val, dict) and "value" in val:
                        block_map[label] = MemoryBlock(label=label, **val)
                    elif isinstance(val, str):
                        block_map[label] = MemoryBlock(label=label, value=val)

                return WorkingMemoryRecord(
                    user_id=row["user_id"],
                    blocks=WorkingMemoryBlocks.from_block_map(block_map),
                    updated_at=row["updated_at"]
                )
        except Exception as e:
            logger.error(f"Failed to fetch working memory blocks: {e}")
            raise StorageOperationError(f"Working memory fetch error: {e}")
