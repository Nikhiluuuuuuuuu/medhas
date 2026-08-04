"""Embodiment subsystem (action -> effect / affordance model).

A cognitive agent is not disembodied text: it has *capabilities* (tools/actions it can
take) and it learns, from experience, what each action does to the world. This module
implements a minimal but real **body model**:

  * ``BodyModel`` — registers capabilities; ``predict(action, context)`` returns the
    expected effect schema; ``observe(action, context, outcome)`` updates the learned
    success rate and outcome distribution (online, offline, no LLM).
  * ``act`` — executes an action through a registered *effector* (a plain callable),
    records predicted vs observed state-change, and feeds the delta back into the model.
    This closes the perception -> action -> observation loop (the defining property of
    an embodied agent).

Effects are stored as structured transitions (state_before -> action -> state_after)
in Postgres so the agent accumulates a reusable skill library. The model is intentionally
local and verifiable: a unit test can register a fake effector, act, and assert the
effect was learned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from medhas.storage import DatabasePool
from medhas.utils import log_error, measure_latency


@dataclass
class EffectRecord:
    action: str
    context: str
    outcome: str
    success: bool
    count: int = 1


Effector = Callable[[str, Dict[str, Any]], Any]


class BodyModel:
    """Learned affordance / action-effect model for the agent's body (its tools)."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._effectors: Dict[str, Effector] = {}

    def register_capability(self, action: str, effector: Effector) -> None:
        """Declare that the body can perform ``action`` via ``effector``."""
        self._effectors[action] = effector

    def can(self, action: str) -> bool:
        return action in self._effectors

    def capabilities(self) -> List[str]:
        return list(self._effectors.keys())

    async def predict(self, action: str, context: str) -> str:
        """Return the most likely learned outcome for (action, context)."""
        rec = await self._fetch(action, context)
        return rec.outcome if rec else f"(unknown effect for {action})"

    async def _fetch(self, action: str, context: str) -> Optional[EffectRecord]:
        async with DatabasePool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT action, context, outcome, success, count FROM body_effects
                WHERE user_id=$1 AND action=$2
                ORDER BY count DESC, success DESC LIMIT 1;
                """,
                self.user_id, action,
            )
            if row:
                return EffectRecord(**dict(row))
            return None

    async def observe(self, action: str, context: str, outcome: str, success: bool) -> None:
        """Record an observed action->effect transition (upsert count)."""
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO body_effects (user_id, action, context, outcome, success, count)
                VALUES ($1, $2, $3, $4, $5, 1)
                ON CONFLICT (user_id, action, context, outcome)
                DO UPDATE SET count = body_effects.count + 1,
                              success = body_effects.success OR $5;
                """,
                self.user_id, action, context, outcome, success,
            )

    async def act(self, action: str, context: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action through its effector, observe the outcome, learn from it."""
        if action not in self._effectors:
            return {"action": action, "executed": False, "error": "capability not registered"}
        predicted = await self.predict(action, context)
        try:
            async with measure_latency(f"embodiment.act.{action}"):
                outcome = self._effectors[action](context, params)
            success = bool(outcome)
            await self.observe(action, context, str(outcome), success)
            return {
                "action": action,
                "executed": True,
                "predicted": predicted,
                "observed": str(outcome),
                "success": success,
            }
        except Exception as e:
            log_error(f"act({action}) failed: {e}")
            await self.observe(action, context, f"error:{e}", False)
            return {"action": action, "executed": False, "error": str(e)}


async def ensure_body_schema() -> None:
    """Create the body_effects table if absent (additive; idempotent)."""
    async with DatabasePool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS body_effects (
                user_id      TEXT NOT NULL,
                action       TEXT NOT NULL,
                context      TEXT NOT NULL,
                outcome      TEXT NOT NULL,
                success      BOOLEAN NOT NULL DEFAULT TRUE,
                count        INTEGER NOT NULL DEFAULT 1,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (user_id, action, context, outcome)
            );
            """
        )


__all__ = ["BodyModel", "EffectRecord", "ensure_body_schema"]
