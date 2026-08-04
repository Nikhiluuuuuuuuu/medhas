"""Layer 1 (Convex): Create a new chat session."""

import json
from typing import Dict, Any, Optional
from medhas.storage import DatabasePool
from medhas.schemas import SessionSchema
from medhas.utils import measure_latency, log_session, log_error
from medhas.core.exceptions import StorageOperationError

async def create_session(user_id: str, metadata: Optional[Dict[str, Any]] = None) -> SessionSchema:
    """Create a new session record in PostgreSQL."""
    meta = metadata or {}
    async with measure_latency("memory.session.create_session"):
        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO sessions (user_id, metadata)
                    VALUES ($1, $2::jsonb)
                    RETURNING id, user_id, metadata, created_at, updated_at;
                    """,
                    user_id,
                    json.dumps(meta)
                )
                assert row is not None, "Failed to insert session row"
                session = SessionSchema(
                    id=row["id"],
                    user_id=row["user_id"],
                    metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                log_session(f"Created session [bold white]{session.id}[/bold white] for user [bold white]{user_id}[/bold white]")
                return session
        except Exception as e:
            log_error(f"Failed to create session: {e}")
            raise StorageOperationError(f"Session creation error: {e}")

async def ensure_session_exists(session_id: Any, user_id: str) -> None:
    """Ensure a session ID exists in PostgreSQL, inserting it if missing."""
    async with measure_latency("memory.session.ensure_session_exists"):
        try:
            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO sessions (id, user_id, metadata)
                    VALUES ($1, $2, '{}'::jsonb)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    session_id,
                    user_id
                )
        except Exception as e:
            log_error(f"Failed to ensure session exists: {e}")

