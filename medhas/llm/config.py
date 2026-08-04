"""LLM configuration models (provider-agnostic, Pydantic).

A single ``LLMConfig`` drives any provider. Nothing here mentions Groq, OpenAI, etc.
by name beyond the free-form ``provider`` string resolved by ``LLMFactory``.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Provider-agnostic LLM configuration.

    Examples
    --------
    OpenAI:
        LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-...")
    Groq (no hard-coded Groq code — just an OpenAI-compatible endpoint):
        LLMConfig(provider="openai", model="llama-3.3-70b-versatile",
                  api_key="gsk_...", base_url="https://api.groq.com/openai/v1")
    Ollama:
        LLMConfig(provider="ollama", model="llama3.1")
    LiteLLM (100+ providers via one path):
        LLMConfig(provider="litellm", model="anthropic/claude-3-5-sonnet")
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: float = 60.0
    top_p: Optional[float] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    def merged(self, **overrides) -> "LLMConfig":
        return self.model_copy(update={k: v for k, v in overrides.items() if v is not None})
