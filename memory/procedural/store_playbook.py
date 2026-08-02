"""AGI Procedural Memory: Store executable skill playbooks and task trajectories."""

import json
from typing import List, Dict, Any, Optional
from infrastructure.db import DatabasePool
from utils import measure_latency, log_working, log_error

async def store_skill_playbook(
    user_id: str,
    task_pattern: str,
    steps: List[str],
    success_rate: float = 1.0
) -> Dict[str, Any]:
    """Store procedural skill playbook for instant workflow execution across multi-playbook map."""
    async with measure_latency("memory.procedural.store_skill_playbook"):
        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow("SELECT blocks FROM working_memory WHERE user_id = $1;", user_id)
                blocks = {}
                if row and row["blocks"]:
                    blocks = json.loads(row["blocks"]) if isinstance(row["blocks"], str) else dict(row["blocks"])
                
                playbooks = blocks.get("procedural_playbooks", {})
                if not isinstance(playbooks, dict):
                    playbooks = {}
                
                single_pb = {"task": task_pattern, "steps": steps, "success_rate": success_rate}
                playbooks[task_pattern.lower().strip()] = single_pb
                blocks["procedural_playbooks"] = playbooks
                blocks["procedural_playbook"] = single_pb

                await conn.execute(
                    """
                    INSERT INTO working_memory (user_id, blocks)
                    VALUES ($1, $2::jsonb)
                    ON CONFLICT (user_id)
                    DO UPDATE SET blocks = $2::jsonb;
                    """,
                    user_id,
                    json.dumps(blocks)
                )
                log_working(f"⚙️ [PROCEDURAL SKILL] Stored skill playbook for task pattern: [bold white]'{task_pattern}'[/bold white]")
                return single_pb
        except Exception as e:
            log_error(f"Store skill playbook error: {e}")
            return {"task": task_pattern, "steps": steps, "success_rate": success_rate}
