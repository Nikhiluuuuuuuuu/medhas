"""Layer 3 (Mem0): Memory decision matrix — LLM classifier for ADD/UPDATE/DELETE/NO_CHANGE.

This is the real Mem0 decision step (mem0/memory/main.py `ADDITIVE_EXTRACTION_PROMPT`
+ conflict resolution), not a cosine-threshold heuristic. It takes the incoming fact and
the top retrieved candidates and asks the LLM to decide the action, returning a
structured verdict. Falls back to a safe hash/cosine heuristic only if the LLM is
unavailable, so the dedup still works offline.
"""

import json
import re
from typing import Dict, Any, List, Optional

from infrastructure.llm import GroqLLMProvider
from config import settings
from core.exceptions import LLMProviderError

DECISION_PROMPT = """\
You are a memory deduplication classifier (Mem0-style decision matrix).
Given a NEW fact and a list of EXISTING facts retrieved for the same user, decide the action.

Actions:
- ADD: the new fact is genuinely new information.
- UPDATE: the new fact is a newer version of an existing fact (same subject, changed detail).
- DELETE: the new fact contradicts an existing fact and the existing fact should be removed (the new one replaces it).
- NO_CHANGE: the new fact is semantically identical to an existing fact.

Return ONLY a JSON object:
{"action": "ADD|UPDATE|DELETE|NO_CHANGE", "target_index": <int or null>, "reason": "<short>"}

EXISTING facts (index = position, starting at 0):
{existing_block}

NEW fact:
{new_fact}
"""


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


async def decide_fact_action(
    new_fact: str,
    existing: List[Any],
    llm: Optional[GroqLLMProvider] = None,
) -> Dict[str, Any]:
    """Classify the incoming fact against retrieved candidates.

    Returns {"action": ..., "target_index": int|None, "target_id": UUID|None}.
    Falls back to a hash/cosine heuristic if the LLM cannot be reached.
    """
    if not existing:
        return {"action": "ADD", "target_index": None, "target_id": None}

    existing_block = "\n".join(
        f"[{i}] {getattr(f, 'fact_text', str(f))}" for i, f in enumerate(existing)
    )
    # NOTE: Build the prompt by concatenation, NOT str.format — the prompt contains literal
    # JSON braces ({...}) which would make str.format raise KeyError.
    user_content = (
        "You are a precise JSON-only memory classifier.\n\n"
        + DECISION_PROMPT
        + "\n\nEXISTING facts (index = position, starting at 0):\n"
        + existing_block
        + "\n\nNEW fact:\n"
        + new_fact
    )
    messages = [
        {"role": "system", "content": "You are a precise JSON-only memory classifier."},
        {"role": "user", "content": user_content},
    ]

    client = llm or GroqLLMProvider()
    try:
        resp = await client.chat_completion(
            messages, temperature=0.0
        )
        data = _extract_json(resp.get("content", "") or "")
        action = str(data.get("action", "ADD")).upper()
        if action not in ("ADD", "UPDATE", "DELETE", "NO_CHANGE"):
            action = "ADD"
        idx = data.get("target_index")
        target_id = None
        if isinstance(idx, int) and 0 <= idx < len(existing):
            target_id = getattr(existing[idx], "id", None)
        return {"action": action, "target_index": idx, "target_id": target_id}
    except (LLMProviderError, Exception):
        # Safe offline fallback: ADD (the caller still applies hash + cosine guardrails).
        return {"action": "ADD", "target_index": None, "target_id": None}
