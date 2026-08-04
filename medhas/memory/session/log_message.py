"""Layer 1 (Convex): Append message to immutable session audit log."""

import json
from uuid import UUID
from typing import Dict, Any, Optional
from medhas.storage import DatabasePool
from medhas.schemas import MessageSchema
from medhas.utils import measure_latency, logger
from medhas.core.exceptions import StorageOperationError

async def log_message(
    session_id: UUID,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> MessageSchema:
    """Log an immutable chat message turn."""
    meta = metadata or {}
    async with measure_latency("memory.session.log_message"):
        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO messages (session_id, role, content, metadata)
                    VALUES ($1, $2, $3, $4::jsonb)
                    RETURNING id, session_id, role, content, metadata, created_at;
                    """,
                    session_id,
                    role,
                    content,
                    json.dumps(meta)
                )
                assert row is not None, "Failed to insert message row"
                message = MessageSchema(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"]),
                    created_at=row["created_at"]
                )
                return message
        except Exception as e:
            logger.error(f"Failed to log message: {e}")
            raise StorageOperationError(f"Log message error: {e}")
