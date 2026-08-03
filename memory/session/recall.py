"""Layer 1 (Convex/Letta): Conversation recall — search recent message history.

Letta keeps a separate *recall* tier (recent conversation messages, chronologically
searchable) distinct from the *archival* cold store. This module implements recall:
semantic + recency search over the immutable chat transcript, so the agent can pull
relevant past turns on demand (Letta conversation_search / recall_memory).
"""

import json
from typing import List, Optional
from uuid import UUID

from infrastructure.db import DatabasePool
from infrastructure.llm import FastEmbeddingProvider
from schemas import MessageSchema
from config import settings
from utils import measure_latency, logger
from core.exceptions import StorageOperationError

embedder = FastEmbeddingProvider()


async def recall_conversation(
    session_id: UUID,
    query: Optional[str] = None,
    limit: int = 5,
) -> List[MessageSchema]:
    """Recall the most relevant recent messages for a session.

    If `query` is given, ranks by vector similarity to the query (semantic recall);
    otherwise returns the most recent messages (chronological recall).
    """
    async with measure_latency("memory.session.recall_conversation"):
        try:
            async with DatabasePool.acquire() as conn:
                if query:
                    try:
                        embedding = await embedder.embed_text(query)
                        vec = f"[{','.join(str(x) for x in embedding)}]"
                        rows = await conn.fetch(
                            """
                            SELECT id, session_id, role, content, metadata, created_at,
                                   1 - (embedding <=> $3::vector) AS sim
                            FROM messages
                            WHERE session_id = $1 AND embedding IS NOT NULL
                            ORDER BY embedding <=> $3::vector ASC
                            LIMIT $2;
                            """,
                            session_id, limit, vec,
                        )
                    except Exception:
                        # No embedding column populated yet → fall back to chronological recall.
                        rows = await conn.fetch(
                            """
                            SELECT id, session_id, role, content, metadata, created_at
                            FROM (
                                SELECT id, session_id, role, content, metadata, created_at
                                FROM messages WHERE session_id = $1
                                ORDER BY created_at DESC LIMIT $2
                            ) sub ORDER BY created_at ASC;
                            """,
                            session_id, limit,
                        )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT id, session_id, role, content, metadata, created_at
                        FROM (
                            SELECT id, session_id, role, content, metadata, created_at
                            FROM messages WHERE session_id = $1
                            ORDER BY created_at DESC LIMIT $2
                        ) sub ORDER BY created_at ASC;
                        """,
                        session_id, limit,
                    )
                return [
                    MessageSchema(
                        id=row["id"],
                        session_id=row["session_id"],
                        role=row["role"],
                        content=row["content"],
                        metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"]),
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to recall conversation: {e}")
            raise StorageOperationError(f"Recall conversation error: {e}")
