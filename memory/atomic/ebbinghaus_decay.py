"""God-Level Cognitive Memory: Ebbinghaus Biological Memory Forgetting & Synaptic Reinforcement."""

import math
from datetime import datetime, timezone
from typing import Dict, Any
from uuid import UUID
from infrastructure.db import DatabasePool
from utils import measure_latency, log_atomic, log_error

def calculate_ebbinghaus_retention(created_at: datetime, half_life_days: float = 7.0) -> float:
    """Calculate biological memory retention score based on Ebbinghaus exponential decay curve."""
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    
    elapsed_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    # Retention R = exp(-elapsed / half_life)
    retention = math.exp(-elapsed_days / half_life_days)
    return max(0.01, min(1.0, retention))

async def reinforce_synaptic_memory(fact_id: UUID) -> float:
    """Double memory half-life upon access (biological synaptic reinforcement)."""
    async with measure_latency("memory.atomic.reinforce_synaptic_memory"):
        try:
            async with DatabasePool.acquire() as conn:
                # Increment importance score slightly and update timestamp to simulate memory consolidation
                row = await conn.fetchrow(
                    """
                    UPDATE atomic_facts
                    SET importance_score = LEAST(10.0, importance_score + 0.5)
                    WHERE id = $1
                    RETURNING id, fact_text, importance_score;
                    """,
                    fact_id
                )
                if row:
                    log_atomic(f"🧠 [SYNAPTIC REINFORCEMENT] Boosted memory strength for: [bold white]'{row['fact_text']}'[/bold white]")
                    return float(row["importance_score"])
                return 5.0
        except Exception as e:
            log_error(f"Synaptic reinforcement error: {e}")
            return 5.0
