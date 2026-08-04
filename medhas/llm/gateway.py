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
from .errors import LLMConfigError, LLMRateLimitError, LLMConnectionError, LLMError
from .router import LLMRouter, EXTRACTION, RESOLUTION, REASONING, SYNTHESIS
from medhas.utils import log_error

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


# Graceful-degradation fallback used when the LLM is unavailable (rate-limited,
# offline, key missing). Returns a provider-shaped dict so call sites don't crash.
_LLM_UNAVAILABLE_FALLBACK = {
    "content": (
        "I'm temporarily unable to reach the language model (rate limit or connection "
        "error). Your message was received and stored; please try again in a moment."
    ),
    "model": "unavailable",
    "finish_reason": "error",
    "raw": None,
}


async def safe_chat_completion(messages, *, task: Optional[str] = None,
                               max_retries: int = 1, **kwargs):
    """chat_completion with graceful degradation.

    - On LLMRateLimitError / LLMConnectionError, retries once on the fast/extraction model.
    - If the fallback also fails, returns a user-facing fallback dict instead of raising,
      so a provider outage never crashes the request (e.g. the ASGI app).
    """
    errors = (LLMRateLimitError, LLMConnectionError, LLMError)
    try:
        return await get_llm(task).chat_completion(messages, **kwargs)
    except errors:
        # Retry once on the cheap/fast model (often on a separate quota / lower cost).
        if max_retries > 0:
            try:
                return await get_extractor().chat_completion(messages, **kwargs)
            except errors:
                pass
        log_error("LLM unavailable after retry; returning graceful fallback response.")
        return dict(_LLM_UNAVAILABLE_FALLBACK)
