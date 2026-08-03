"""Layer 4 (Zep/Graphiti): Episodes resolution layer.

Graphiti anchors every extracted fact/edge to an *Episode* (a timestamped, sourced
unit of raw input — a message, a document, a JSON blob). The episode carries
`reference_time`, `source` (EpisodeType) and `source_description`, and is the
provenance anchor for all derived graph elements. Graphiti's public API is
``Graphiti.add_episode(name, episode_body, source_description, reference_time,
source=EpisodeType.message, group_id=..., uuid=...)`` (graphiti_core/graphiti.py:980).

Medhas already persists episodes (async_extractor anchors each turn as an episode),
but had no public resolution API. This module mirrors Graphiti's add_episode
signature so episodes can be created, fetched, and linked to derived memories.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone

from infrastructure.db import DatabasePool
from utils import measure_latency, log_graph, log_error
from core.exceptions import StorageOperationError


class EpisodeType(str, Enum):
    """Graphiti episode source types (graphiti_core/nodes/episode.py)."""

    message = "message"
    text = "text"
    json = "json"
    episode = "episode"


async def add_episode(
    user_id: str,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: Optional[datetime] = None,
    source: EpisodeType = EpisodeType.message,
    group_id: Optional[str] = None,
    episode_uuid: Optional[UUID] = None,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None,
) -> UUID:
    """Create and persist an episode (Graphiti add_episode analogue).

    Returns the episode id. `reference_time` defaults to now (UTC) — Graphiti uses
    it as the temporal anchor for bi-temporal edge validity.
    """
    ref = reference_time or datetime.now(timezone.utc)
    ep_id = episode_uuid or uuid4()
    async with measure_latency("memory.episodes.add_episode"):
        try:
            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO episodes
                        (id, user_id, session_id, agent_id, content, source, reference_time, group_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        source_description = EXCLUDED.source_description,
                        reference_time = EXCLUDED.reference_time;
                    """,
                    ep_id, user_id, session_id, agent_id, episode_body,
                    source.value, ref, group_id or source_description,
                )
                # Backfill source_description into content-adjacent column for query convenience.
                await conn.execute(
                    "UPDATE episodes SET source_description = $2 WHERE id = $1;",
                    ep_id, source_description,
                )
                log_graph(f"Episode anchored: '{name}' ({source.value}) ref={ref.isoformat()}")
                return ep_id
        except Exception as e:
            log_error(f"Failed to add episode: {e}")
            raise StorageOperationError(f"Add episode error: {e}")


async def get_episode(episode_id: UUID) -> Optional[Dict[str, Any]]:
    """Fetch a single episode by id (Graphiti get_episode analogue)."""
    async with measure_latency("memory.episodes.get_episode"):
        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, user_id, session_id, agent_id, content, source,
                           source_description, reference_time, created_at
                    FROM episodes WHERE id = $1;
                    """,
                    episode_id,
                )
                return dict(row) if row else None
        except Exception as e:
            log_error(f"Failed to get episode: {e}")
            return None


async def get_episodes(
    user_id: str,
    limit: int = 20,
    source: Optional[EpisodeType] = None,
    group_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List episodes for a user, newest first (Graphiti episode history)."""
    async with measure_latency("memory.episodes.get_episodes"):
        try:
            async with DatabasePool.acquire() as conn:
                where = ["user_id = $1"]
                params: List[Any] = [user_id]
                p = 2
                if source is not None:
                    where.append(f"source = ${p}")
                    params.append(source.value)
                    p += 1
                if group_id is not None:
                    where.append(f"group_id = ${p}")
                    params.append(group_id)
                    p += 1
                params.append(limit)
                rows = await conn.fetch(
                    f"""
                    SELECT id, user_id, session_id, agent_id, content, source,
                           source_description, reference_time, created_at
                    FROM episodes
                    WHERE {' AND '.join(where)}
                    ORDER BY reference_time DESC
                    LIMIT ${p};
                    """,
                    *params,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            log_error(f"Failed to list episodes: {e}")
            return []
