"""Regression tests for provider-agnostic base_url injection (issue: empty base_url
caused 'Request URL is missing an http:// or https:// protocol' when provider='groq'
was configured with only an API key).

These run offline — they assert the *configuration wiring* (the client is constructed
with a valid http(s) base_url), not a live call.
"""
import os
import sys

import pytest

# Ensure repo root on path so `medhas` resolves when run via pytest discovery.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def _make_config(**overrides):
    from medhas.llm.config import LLMConfig
    cfg = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk-test")
    return cfg.merged(**overrides) if overrides else cfg


def test_groq_alias_injects_base_url_with_key_only():
    """provider='groq' + api_key and NO explicit base_url -> factory injects Groq endpoint."""
    from medhas.llm.factory import create_llm
    from medhas.llm.providers.openai_compatible import OpenAICompatibleLLM

    llm = create_llm(_make_config())  # base_url defaults to None
    assert isinstance(llm, OpenAICompatibleLLM)
    # The AsyncOpenAI client must have received a valid http(s) base_url.
    base = str(llm._client.base_url)
    assert base.startswith("http://") or base.startswith("https://"), base
    assert "groq.com" in base, base


def test_empty_base_url_string_is_treated_as_missing():
    """An explicit empty-string LLM_BASE_URL must still resolve to the provider endpoint."""
    from medhas.llm.factory import create_llm

    llm = create_llm(_make_config(base_url="  "))
    base = str(llm._client.base_url)
    assert "groq.com" in base, base


def test_router_from_settings_injects_groq_when_base_url_empty():
    """LLMRouter.from_settings with LLM_BASE_URL='' (the buggy default) must still work."""

    class _FakeSettings:
        LLM_PROVIDER = "groq"
        LLM_MODEL = "llama-3.3-70b-versatile"
        LLM_API_KEY = "gsk-test"
        LLM_BASE_URL = ""  # the exact misconfiguration from the bug report
        LLM_FAST_MODEL = "llama-3.1-8b-instant"
        LLM_TEMPERATURE = 0.1
        LLM_MAX_TOKENS = 2048
        LLM_TIMEOUT = 60.0

    from medhas.llm.router import LLMRouter
    router = LLMRouter.from_settings(_FakeSettings())
    base = str(router.primary._client.base_url)
    assert "groq.com" in base, base


def test_explicit_base_url_is_respected():
    """If the user supplies a custom OpenAI-compatible base_url, it is NOT overridden."""
    from medhas.llm.factory import create_llm

    llm = create_llm(_make_config(base_url="https://my-proxy.example.com/v1"))
    base = str(llm._client.base_url)
    assert "my-proxy.example.com" in base, base


def test_missing_base_url_raises_clear_error():
    """Direct OpenAICompatibleLLM construction with no base_url fails fast with guidance."""
    from medhas.llm.config import LLMConfig
    from medhas.llm.errors import LLMConfigError
    from medhas.llm.providers.openai_compatible import OpenAICompatibleLLM

    with pytest.raises(LLMConfigError):
        OpenAICompatibleLLM(LLMConfig(provider="openai", api_key="x", base_url=""))
