"""E7 (runtime) / E36 — Background consolidation scheduler + rehearsal buffer.

E7: an idle-triggered and interval-triggered background worker that runs the
    consolidation cycle without blocking the hot path ("sleep" for the agent).
E36: a rehearsal buffer that periodically re-activates important-but-unaccessed
     memories so they are not lost to decay (catastrophic-forgetting guard).
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from medhas.storage import DatabasePool
from medhas.utils import log_atomic, log_error, measure_latency

DEFAULT_INTERVAL_SECONDS = 900     # periodic consolidation
DEFAULT_IDLE_SECONDS = 120         # consider a user idle after this long


class ConsolidationScheduler:
    """Idle + periodic background consolidation worker (E7)."""

    def __init__(
        self,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        idle_seconds: int = DEFAULT_IDLE_SECONDS,
    ) -> None:
        self.interval = interval_seconds
        self.idle_seconds = idle_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_activity: Dict[str, datetime] = {}
        self.last_run: Dict[str, datetime] = {}

    # -- activity tracking ------------------------------------------------
    def touch(self, user_id: str) -> None:
        """Record user activity; consolidation waits for idleness."""
        self._last_activity[user_id] = datetime.now(timezone.utc)

    def is_idle(self, user_id: str) -> bool:
        last = self._last_activity.get(user_id)
        if last is None:
            return True
        return (datetime.now(timezone.utc) - last).total_seconds() >= self.idle_seconds

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log_atomic(f"E7 consolidation scheduler started (interval={self.interval}s)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(f"Consolidation scheduler tick failed: {e}")

    async def run_once(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Run consolidation for one user, or every idle user with recent activity."""
        from medhas.memory.operations.consolidation import run_consolidation

        results: Dict[str, Any] = {}
        users = [user_id] if user_id else await self._active_users()
        for uid in users:
            if user_id is None and not self.is_idle(uid):
                continue
            try:
                results[uid] = await run_consolidation(uid)
                self.last_run[uid] = datetime.now(timezone.utc)
                await rehearse(uid)
            except Exception as e:
                log_error(f"Consolidation failed for {uid}: {e}")
                results[uid] = {"status": "error", "message": str(e)}
        return results

    async def _active_users(self, limit: int = 50) -> List[str]:
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT user_id FROM atomic_facts
                    WHERE is_active = TRUE AND created_at > NOW() - INTERVAL '30 days'
                    LIMIT $1;
                    """,
                    limit,
                )
            return [r["user_id"] for r in rows]
        except Exception as e:
            log_error(f"_active_users failed: {e}")
            return []


#: process-wide scheduler instance
scheduler = ConsolidationScheduler()


# --------------------------------------------------------- E36 rehearsal

async def rehearse(user_id: str, limit: int = 10) -> Dict[str, Any]:
    """Re-activate important memories that have not been accessed recently.

    Mirrors experience replay: without this, valuable-but-quiet memories decay out
    even though they were never wrong — the catastrophic-forgetting failure mode.
    """
    async with measure_latency("agi.scheduler.rehearse"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, fact_text FROM atomic_facts
                    WHERE user_id = $1 AND is_active = TRUE
                      AND importance_score >= 6.0
                      AND (last_accessed_at IS NULL OR last_accessed_at < NOW() - INTERVAL '7 days')
                    ORDER BY importance_score DESC, last_accessed_at ASC NULLS FIRST
                    LIMIT $2;
                    """,
                    user_id, limit,
                )
                rehearsed = 0
                for r in rows:
                    await conn.execute(
                        """
                        INSERT INTO rehearsal_buffer (user_id, fact_id, last_rehearsed_at, rehearsal_count)
                        VALUES ($1,$2,CURRENT_TIMESTAMP,1)
                        ON CONFLICT (user_id, fact_id) DO UPDATE SET
                            last_rehearsed_at = CURRENT_TIMESTAMP,
                            rehearsal_count = rehearsal_buffer.rehearsal_count + 1;
                        """,
                        user_id, r["id"],
                    )
                    # Reactivation slows decay without inflating importance unboundedly.
                    await conn.execute(
                        """
                        UPDATE atomic_facts
                        SET last_accessed_at = CURRENT_TIMESTAMP,
                            decay_half_life_days = LEAST(90.0, decay_half_life_days * 1.5)
                        WHERE id = $1;
                        """,
                        r["id"],
                    )
                    rehearsed += 1
            if rehearsed:
                log_atomic(f"E36 rehearsed {rehearsed} important memories")
            return {"rehearsed": rehearsed}
        except Exception as e:
            log_error(f"rehearse failed: {e}")
            return {"rehearsed": 0, "error": str(e)}
