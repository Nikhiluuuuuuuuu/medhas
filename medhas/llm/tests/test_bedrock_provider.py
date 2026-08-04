"""Offline tests for the Amazon Bedrock provider (no AWS calls)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def _make_config(**overrides):
    from medhas.llm.config import LLMConfig
    cfg = LLMConfig(provider="bedrock", model="claude-3-5-sonnet", api_key="AKIA_TEST", base_url="us-east-1")
    return cfg.merged(**overrides) if overrides else cfg


def test_bedrock_model_alias_resolution():
    from medhas.llm.providers.bedrock import _resolve_model_id
    assert _resolve_model_id("claude-3-5-sonnet").startswith("anthropic.claude-3-5-sonnet")
    assert _resolve_model_id("already.full.id") == "already.full.id"


def test_bedrock_message_mapping():
    from medhas.llm.providers.bedrock import _to_bedrock_messages
    system, msgs = _to_bedrock_messages([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert system == "be brief"
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == [{"text": "hi"}]
    assert msgs[1]["role"] == "assistant"


def test_bedrock_constructs_and_calls_via_mock(monkeypatch):
    """Verify the provider maps config -> Converse params and parses the response,
    using a stub boto3 client so no network/AWS call happens."""
    from medhas.llm.providers import BedrockLLM
    import sys
    import types

    calls = {}
    stub = types.ModuleType("boto3")

    class _StubClient:
        def converse(self, **params):
            calls["params"] = params
            return {
                "output": {"message": {"content": [{"text": "bedrock reply"}]}},
                "stopReason": "end_turn",
            }

    stub.client = lambda *a, **k: _StubClient()
    monkeypatch.setitem(sys.modules, "boto3", stub)

    llm = BedrockLLM(_make_config())
    import asyncio
    resp = asyncio.run(llm.acompletion([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]))
    assert resp["content"] == "bedrock reply"
    # model id resolved from alias + region passed through
    assert calls["params"]["modelId"].startswith("anthropic.claude-3-5-sonnet")


def test_bedrock_rate_limit_normalised(monkeypatch):
    from medhas.llm.providers import BedrockLLM
    from medhas.llm.errors import LLMRateLimitError
    import sys
    import types

    stub = types.ModuleType("boto3")

    class _Boom:
        def converse(self, **params):
            raise RuntimeError("ThrottlingException: rate limit (429) exceeded")

    stub.client = lambda *a, **k: _Boom()
    monkeypatch.setitem(sys.modules, "boto3", stub)

    llm = BedrockLLM(_make_config())
    import asyncio
    with pytest.raises(LLMRateLimitError):
        asyncio.run(llm.acompletion([{"role": "user", "content": "q"}]))
