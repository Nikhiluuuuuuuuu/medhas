"""Regression tests for graceful LLM degradation (no crash on 429 / connection error).

The 429 the user hit bubbled up as an unhandled ASGI exception. safe_chat_completion must
return a user-facing fallback dict instead of raising, so a provider outage never crashes
the request.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def test_safe_chat_completion_returns_fallback_on_rate_limit(monkeypatch):
    from medhas.llm import gateway

    class _Boom:
        async def chat_completion(self, *a, **k):
            from medhas.llm.errors import LLMRateLimitError
            raise LLMRateLimitError("429 rate limited")

    # Both primary and fallback raise -> must degrade to a fallback dict, not raise.
    monkeypatch.setattr(gateway, "get_llm", lambda task=None: _Boom())
    monkeypatch.setattr(gateway, "get_extractor", lambda: _Boom())

    import asyncio
    result = asyncio.run(gateway.safe_chat_completion([{"role": "user", "content": "hi"}]))
    assert isinstance(result, dict)
    assert result.get("model") == "unavailable"
    assert result.get("content") and "unable to reach" in result["content"].lower()


def test_safe_chat_completion_passes_through_on_success(monkeypatch):
    from medhas.llm import gateway

    class _Ok:
        async def chat_completion(self, *a, **k):
            return {"content": "ok", "model": "groq/llama"}

    monkeypatch.setattr(gateway, "get_llm", lambda task=None: _Ok())

    import asyncio
    result = asyncio.run(gateway.safe_chat_completion([{"role": "user", "content": "hi"}]))
    assert result["content"] == "ok"


def test_get_embedding_dimension_returns_int():
    from medhas.embeddings.embedding_provider import get_embedding_dimension
    dim = get_embedding_dimension()
    assert isinstance(dim, int) and dim > 0
