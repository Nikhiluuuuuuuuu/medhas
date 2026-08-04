"""Structured-output helper (provider-agnostic).

Uses JSON-mode where available and falls back to prompt-injected JSON parsing. Mirrors
cognee's ``litellm_instructor`` idea: one function that turns a Pydantic model / JSON
schema into a typed response regardless of which provider is underneath.
"""
import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .base import BaseLLM, Message
from .errors import LLMError

T = TypeVar("T", bound=BaseModel)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    # strip ```json fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    match = _JSON_RE.search(text)
    if not match:
        raise LLMError(f"LLM did not return parseable JSON: {text[:200]!r}")
    return json.loads(match.group(0))


async def generate_structured_default(llm: BaseLLM, messages: List[Message],
                                      schema: Dict[str, Any],
                                      **kwargs: Any) -> Dict[str, Any]:
    """Ask the provider for JSON matching ``schema`` and parse it."""
    system = {
        "role": "system",
        "content": (
            "You MUST respond with a single valid JSON object and nothing else, "
            f"matching this JSON schema:\n{json.dumps(schema, indent=2)}"
        ),
    }
    cleaned = [system] + [m for m in messages if m.get("role") != "system"]
    kwargs.setdefault("temperature", 0.0)
    try:
        kwargs["response_format"] = {"type": "json_object"}
    except TypeError:
        pass
    out = await llm.acompletion(cleaned, **kwargs)
    return _coerce_json(out.get("content", ""))


async def generate_structured(llm: BaseLLM, messages: List[Message],
                              model: Type[T], **kwargs: Any) -> T:
    """Typed variant: returns a validated Pydantic instance."""
    data = await generate_structured_default(llm, messages, model.model_json_schema(), **kwargs)
    return model(**data)
