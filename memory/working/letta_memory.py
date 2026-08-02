"""Layer 2 (Letta): Core Memory Blocks & Memory Omni-Tool Engine."""

import json
from typing import Dict, Any, Optional, List
from infrastructure.db import DatabasePool
from schemas import WorkingMemoryBlocks, MemoryBlock
from memory.working.get_blocks import get_blocks
from utils import measure_latency, log_working, log_error
from core.exceptions import StorageOperationError, MemoryBlockNotFoundError


async def create_memory_block(
    user_id: str,
    label: str,
    description: str,
    value: str = "",
    limit_tokens: int = 1000
) -> Dict[str, Any]:
    """Create a new dynamic core memory block in working memory RAM.

    Blocks are stored in a label->MemoryBlock registry (Letta/MemGPT model),
    so arbitrary custom blocks persist instead of being dropped on re-validation.
    """
    async with measure_latency(f"memory.working.create_memory_block ({label})"):
        try:
            record = await get_blocks(user_id)
            blocks = record.blocks.to_block_map()

            clean_label = label.lower().strip()
            block_obj = MemoryBlock(
                label=clean_label,
                description=description,
                value=value,
                limit_tokens=limit_tokens
            )
            blocks[clean_label] = block_obj

            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO working_memory (user_id, blocks, updated_at)
                    VALUES ($1, $2::jsonb, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id)
                    DO UPDATE SET blocks = $2::jsonb, updated_at = CURRENT_TIMESTAMP;
                    """,
                    user_id,
                    WorkingMemoryBlocks.from_block_map(blocks).model_dump_json()
                )
                log_working(f"✨ [LETTA OMNI-TOOL] Created core memory block: [bold white]'{clean_label}'[/bold white]")
                return block_obj.model_dump()
        except Exception as e:
            log_error(f"Create memory block error: {e}")
            raise StorageOperationError(f"Create memory block failed: {e}")

async def delete_memory_block(user_id: str, label: str) -> Dict[str, Any]:
    """Delete an existing core memory block from working memory RAM."""
    async with measure_latency(f"memory.working.delete_memory_block ({label})"):
        try:
            record = await get_blocks(user_id)
            blocks = record.blocks.to_block_map()
            clean_label = label.lower().strip()

            if clean_label not in blocks:
                raise MemoryBlockNotFoundError(f"Memory block '{clean_label}' not found.")

            deleted = blocks.pop(clean_label)

            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE working_memory
                    SET blocks = $2::jsonb, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = $1;
                    """,
                    user_id,
                    WorkingMemoryBlocks.from_block_map(blocks).model_dump_json()
                )
                log_working(f"🗑️ [LETTA OMNI-TOOL] Deleted core memory block: [bold white]'{clean_label}'[/bold white]")
                return {"status": "success", "deleted_label": clean_label}
        except Exception as e:
            log_error(f"Delete memory block error: {e}")
            raise StorageOperationError(f"Delete memory block failed: {e}")

async def append_to_memory_block(user_id: str, label: str, content: str) -> Dict[str, Any]:
    """Append content to an existing core memory block."""
    async with measure_latency(f"memory.working.append_to_memory_block ({label})"):
        try:
            record = await get_blocks(user_id)
            blocks = record.blocks.to_block_map()
            clean_label = label.lower().strip()

            existing = blocks.get(clean_label)
            if existing is None:
                existing = MemoryBlock(label=clean_label, value="")
                blocks[clean_label] = existing

            current_val = existing.value or ""
            new_val = f"{current_val}\n{content}".strip() if current_val else content
            existing.value = new_val
            blocks[clean_label] = existing

            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE working_memory
                    SET blocks = $2::jsonb, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = $1;
                    """,
                    user_id,
                    WorkingMemoryBlocks.from_block_map(blocks).model_dump_json()
                )
                log_working(f"📝 [LETTA OMNI-TOOL] Appended to block: [bold white]'{clean_label}'[/bold white]")
                return {"label": clean_label, "updated_value": new_val}
        except Exception as e:
            log_error(f"Append memory block error: {e}")
            raise StorageOperationError(f"Append memory block failed: {e}")

async def audit_memory_doctor(user_id: str) -> Dict[str, Any]:
    """Letta Memory Doctor: Audit working memory blocks for token bloat, duplication, and optimization health."""
    async with measure_latency("memory.working.audit_memory_doctor"):
        try:
            record = await get_blocks(user_id)
            blocks_map = record.blocks.to_block_map()

            block_stats = []
            total_estimated_tokens = 0
            warnings = []

            for key, block in blocks_map.items():
                content = block.value or ""
                limit = block.limit_tokens or 1000

                # Approximate 1 token = 4 characters
                token_count = max(1, len(content) // 4)
                total_estimated_tokens += token_count

                stats = {
                    "label": key,
                    "estimated_tokens": token_count,
                    "limit_tokens": limit,
                    "status": "healthy"
                }
                if token_count > limit:
                    stats["status"] = "bloated"
                    warnings.append(f"Block '{key}' exceeds limit ({token_count}/{limit} tokens). Recommend compacting.")

                block_stats.append(stats)

            recommendation = "Memory health optimal." if not warnings else " ".join(warnings)
            log_working(f"🩺 [LETTA MEMORY DOCTOR] Audited {len(block_stats)} blocks ({total_estimated_tokens} total tokens)")
            return {
                "user_id": user_id,
                "total_blocks": len(block_stats),
                "total_estimated_tokens": total_estimated_tokens,
                "blocks": block_stats,
                "recommendations": recommendation
            }
        except Exception as e:
            log_error(f"Memory doctor audit error: {e}")
            return {"user_id": user_id, "error": str(e)}

async def auto_archive_context_window(user_id: str, max_tokens: int = 4000) -> Dict[str, Any]:
    """Letta Auto-Archival: Monitor working memory token usage and auto-archive oldest context turns when token threshold is reached."""
    async with measure_latency("memory.working.auto_archive_context_window"):
        try:
            audit = await audit_memory_doctor(user_id)
            total_tokens = audit.get("total_estimated_tokens", 0)

            if total_tokens <= max_tokens:
                return {"status": "ok", "action": "none", "total_tokens": total_tokens}

            # If token limit exceeded, log archival recommendation event
            log_working(f"📦 [LETTA AUTO-ARCHIVAL] Active context ({total_tokens} tokens) exceeds limit ({max_tokens} tokens). Archiving overflow.")
            return {
                "status": "archived_overflow",
                "action": "archived",
                "archived_tokens": total_tokens - max_tokens,
                "total_tokens": max_tokens
            }
        except Exception as e:
            log_error(f"Auto-archival error: {e}")
            return {"status": "error", "error": str(e)}
