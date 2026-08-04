"""LLM gateway — process-wide access to the configured provider router.

Replaces the old hard-coded ``GroqLLMProvider()`` singleton. Call sites now request an LLM
by logical task (``extraction``, ``resolution``, ``reasoning``, ``synthesis``) and receive a
provider-agnostic :class:`~medhas.llm.base.BaseLLM`. The concrete provider is chosen purely
by configuration (``medhas.config.settings``) — no vendor code lives at the call site.
"""
import threading
from typing import Optional

from .base import BaseLLM
from .config import LLMConfig
from .errors import LLMConfigError
from .router import LLMRouter, EXTRACTION, RESOLUTION, REASONING, SYNTHESIS

_router: Optional[LLMRouter] = None
_lock = threading.Lock()


def _build_router() -> LLMRouter:
    # Imported lazily to avoid a settings import cycle at module load.
    from medhas.config import settings
    return LLMRouter.from_settings(settings)


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        with _lock:
            if _router is None:
                _router = _build_router()
    return _router


def set_router(router: LLMRouter) -> None:
    """Override the router (tests / explicit configuration)."""
    global _router
    with _lock:
        _router = router


def get_llm(task: Optional[str] = None) -> BaseLLM:
    """Return a provider-agnostic LLM for ``task`` (or the primary)."""
    if task is None:
        return get_router().primary
    return get_router().for_task(task)


def get_extractor() -> BaseLLM:
    return get_llm(EXTRACTION)


def get_resolver() -> BaseLLM:
    return get_llm(RESOLUTION)


def get_dream_llm() -> BaseLLM:
    return get_llm(REASONING)


def get_synthesis_llm() -> BaseLLM:
    return get_llm(SYNTHESIS)


def reset() -> None:
    """Drop the cached router (e.g. after changing settings)."""
    global _router
    with _lock:
        _router = None
