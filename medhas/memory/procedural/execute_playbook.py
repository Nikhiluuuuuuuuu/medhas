"""AGI Procedural Memory: Fetch and execute skill playbooks."""

import json
from typing import Optional, Dict, Any
from medhas.storage import DatabasePool
from medhas.utils import measure_latency, log_working, log_error

async def get_skill_playbook(user_id: str, task_pattern: str) -> Optional[Dict[str, Any]]:
    """Retrieve stored procedural skill playbook matching task pattern dynamically across all saved playbooks."""
    async with measure_latency("memory.procedural.get_skill_playbook"):
        try:
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow("SELECT blocks FROM working_memory WHERE user_id = $1;", user_id)
                if not row:
                    return None
                
                blocks = json.loads(row["blocks"]) if isinstance(row["blocks"], str) else dict(row["blocks"])
                query_task = task_pattern.lower().strip()

                # Check multi-playbook registry first
                playbooks = blocks.get("procedural_playbooks", {})
                if isinstance(playbooks, dict):
                    for key, pb in playbooks.items():
                        stored_task = pb.get("task", "").lower().strip()
                        if stored_task and (stored_task in query_task or query_task in stored_task or key in query_task):
                            log_working(f"⚡ [PROCEDURAL SKILL MATCH] Found skill playbook for: [bold white]'{task_pattern}'[/bold white]")
                            return pb

                # Fallback check single playbook
                playbook = blocks.get("procedural_playbook")
                if playbook:
                    stored_task = playbook.get("task", "").lower().strip()
                    if stored_task and (stored_task in query_task or query_task in stored_task):
                        log_working(f"⚡ [PROCEDURAL SKILL MATCH] Found skill playbook for: [bold white]'{task_pattern}'[/bold white]")
                        return playbook
                return None
        except Exception as e:
            log_error(f"Get skill playbook error: {e}")
            return None
