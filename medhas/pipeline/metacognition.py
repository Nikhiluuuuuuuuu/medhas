"""AGI Metacognitive Executive Controller: Dual-Process Cognitive Engine (System 1 vs System 2)."""

from typing import Dict, Any, Tuple
from medhas.memory.procedural import get_skill_playbook
from medhas.utils import measure_latency, log_working, log_error

async def evaluate_cognitive_mode(user_id: str, user_message: str) -> Tuple[str, Dict[str, Any]]:
    """Determine System 1 (Fast Path Direct Memory) vs System 2 (Slow Path Multi-Hop RRF + SAN Reasoning)."""
    async with measure_latency("pipeline.metacognition.evaluate_cognitive_mode"):
        # Check if Procedural Skill Playbook exists
        playbook = await get_skill_playbook(user_id, user_message)
        if playbook:
            log_working(f"⚡ [METACOGNITION: SYSTEM 1 FAST PATH] Matched procedural skill playbook: {playbook['task']}")
            return "SYSTEM_1", {"playbook": playbook}

        # Simple greeting or short turn -> System 1 Fast Path
        if len(user_message.split()) <= 3 and any(w in user_message.lower() for w in ["hi", "hello", "hey", "thanks"]):
            log_working("⚡ [METACOGNITION: SYSTEM 1 FAST PATH] Direct heuristic completion")
            return "SYSTEM_1", {}

        # Complex reasoning, domain updates, or mult-hop query -> System 2 Slow Path
        log_working("🧠 [METACOGNITION: SYSTEM 2 SLOW PATH] Deep RRF Hybrid Search + SAN Graph Traversal Enabled")
        return "SYSTEM_2", {}
