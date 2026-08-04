"""Abstract LLM provider base.

Every concrete provider (OpenAI-compatible, Anthropic, Gemini, Ollama, vLLM, LiteLLM)
implements this interface so the rest of the system depends only on ``BaseLLM`` and
never on a specific vendor SDK.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .config import LLMConfig
from .errors import LLMError

# A chat message is a plain dict: {"role": "system"|"user"|"assistant", "content": str}
Message = Dict[str, str]


class BaseLLM(ABC):
    """Common interface for all LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    async def acompletion(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        """Async chat completion.

        Returns a dict with at least ``{"content": str, "model": str}`` plus optional
        ``finish_reason`` / ``raw``. Concrete providers normalise their SDK response
        into this shape.
        """

    async def completion(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        """Sync-style wrapper (still async under the hood)."""
        return await self.acompletion(messages, **kwargs)

    async def generate_structured(self, messages: List[Message], schema: Dict[str, Any],
                                  **kwargs: Any) -> Dict[str, Any]:
        """Best-effort structured output.

        Default implementation injects the JSON schema into a system prompt and asks for
        ``response_format={"type":"json_object"}`` when the provider supports it. Providers
        with native structured output should override this.
        """
        from .structured import generate_structured_default
        return await generate_structured_default(self, messages, schema, **kwargs)

    # Backwards-compatible alias so legacy call sites (`.chat_completion(messages, temperature=...)`)
    # keep working after the Groq-specific client was removed.
    async def chat_completion(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        return await self.acompletion(messages, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} {self.config.provider}:{self.config.model}>"
