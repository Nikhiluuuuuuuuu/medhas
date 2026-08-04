"""LLM factory — config-driven provider registry (mem0 ``LlmFactory`` style).

Maps a free-form ``provider`` string to a concrete ``BaseLLM`` class. Adding a new vendor
is a one-line registry entry — no call-site changes anywhere in the engine.
"""
from .base import BaseLLM
from .config import LLMConfig
from .errors import LLMConfigError
from .providers import (
    AnthropicLLM,
    GeminiLLM,
    LiteLLMLLM,
    OllamaLLM,
    OpenAICompatibleLLM,
)

# provider string -> (class, required-env-var-or-None)
_REGISTRY = {
    "openai": (OpenAICompatibleLLM, None),
    "groq": (OpenAICompatibleLLM, "GROQ_API_KEY"),        # OpenAI-compatible base_url
    "together": (OpenAICompatibleLLM, "TOGETHER_API_KEY"),
    "deepseek": (OpenAICompatibleLLM, "DEEPSEEK_API_KEY"),
    "openrouter": (OpenAICompatibleLLM, "OPENROUTER_API_KEY"),
    "ollama": (OllamaLLM, None),
    "vllm": (OpenAICompatibleLLM, None),
    "lmstudio": (OpenAICompatibleLLM, None),
    "anthropic": (AnthropicLLM, "ANTHROPIC_API_KEY"),
    "gemini": (GeminiLLM, "GEMINI_API_KEY"),
    "litellm": (LiteLLMLLM, None),
}


def register_provider(name: str, cls, *, required_env: str = None) -> None:
    """Extend the registry at runtime (e.g. BYOK providers, like Letta)."""
    _REGISTRY[name.lower()] = (cls, required_env)


def create_llm(config: LLMConfig) -> BaseLLM:
    """Build a provider instance from an :class:`LLMConfig`."""
    entry = _REGISTRY.get(config.provider.lower())
    if entry is None:
        raise LLMConfigError(f"Unknown LLM provider: {config.provider!r}. "
                              f"Known: {sorted(_REGISTRY)}")
    cls, required_env = entry
    if required_env and not config.api_key:
        import os
        if not os.environ.get(required_env):
            raise LLMConfigError(
                f"Provider {config.provider!r} needs {required_env} (or set config.api_key).")
    # OpenAI-compatible aliases need the right base_url unless caller supplied one.
    if cls is OpenAICompatibleLLM and config.base_url is None:
        base_urls = {
            "groq": "https://api.groq.com/openai/v1",
            "together": "https://api.together.xyz/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        if config.provider.lower() in base_urls:
            config = config.merged(base_url=base_urls[config.provider.lower()])
    return cls(config)
