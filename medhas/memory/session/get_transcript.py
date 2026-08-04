"""Layer 1 (Convex): Fetch recent chat history transcript."""

import json
from uuid import UUID
from typing import List
from medhas.storage import DatabasePool
from medhas.schemas import MessageSchema
from medhas.config import settings
from medhas.utils import measure_latency, logger
from medhas.core.exceptions import StorageOperationError

async def get_transcript(session_id: UUID, limit: int = settings.MAX_HISTORICAL_MESSAGES) -> List[MessageSchema]:
    """Retrieve recent chat history messages ordered chronologically."""
    async with measure_latency("memory.session.get_transcript"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, role, content, metadata, created_at
                    FROM (
                        SELECT id, session_id, role, content, metadata, created_at
                        FROM messages
                        WHERE session_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                    ) sub
                    ORDER BY created_at ASC;
                    """,
                    session_id,
                    limit
                )
                return [
                    MessageSchema(
                        id=row["id"],
                        session_id=row["session_id"],
                        role=row["role"],
                        content=row["content"],
                        metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"]),
                        created_at=row["created_at"]
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch transcript: {e}")
            raise StorageOperationError(f"Fetch transcript error: {e}")
