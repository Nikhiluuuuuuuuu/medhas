"""Amazon Bedrock provider (Anthropic / Amazon / Cohere / Mistral models on Bedrock).

Uses the Bedrock ``Converse`` API, which normalises message handling across every model
family hosted on Bedrock, so a single implementation covers Claude, Amazon Titan,
Command (Cohere), Mistral, etc.

Credentials are taken from the standard AWS chain:
  - ``LLM_API_KEY``  -> AWS access key id
  - ``AWS_SECRET_ACCESS_KEY`` (env) -> AWS secret access key
  - ``LLM_BASE_URL`` (optional) -> AWS region (e.g. ``us-east-1``)
If those are unset, boto3 falls back to ~/.aws, environment, or the attached IAM role.
"""
from typing import Any, Dict, List

from ..base import BaseLLM, Message
from ..config import LLMConfig
from ..errors import LLMRateLimitError, LLMTimeoutError, LLMConnectionError, LLMError, LLMConfigError
from ..metrics import measure_latency

# Sensible Bedrock model-id defaults if only a short name is given.
_BEDROCK_MODEL_ALIASES = {
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-v2": "anthropic.claude-v2",
    "titan-text": "amazon.titan-text-express-v1",
    "command-r": "cohere.command-r-plus-v1:0",
    "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
    "mixtral": "mistral.mixtral-8x7b-instruct-v0:1",
}


def _resolve_model_id(model: str) -> str:
    if model in _BEDROCK_MODEL_ALIASES:
        return _BEDROCK_MODEL_ALIASES[model]
    return model


def _to_bedrock_messages(messages: List[Message]):
    """Convert OpenAI-style messages to Bedrock Converse format.

    Bedrock Converse expects user/assistant turns plus a separate system string.
    """
    system = ""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system += content + "\n"
            continue
        bedrock_role = "user" if role in ("user", "tool") else "assistant"
        out.append({"role": bedrock_role, "content": [{"text": content}]})
    return system.strip(), out


class BedrockLLM(BaseLLM):
    """AWS Bedrock via boto3 ``Converse`` API."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            import boto3  # lazy: only required when this provider is actually used
        except ImportError as e:  # pragma: no cover
            raise LLMError("`boto3` is required for BedrockLLM (`pip install boto3`)") from e

        client_kwargs: Dict[str, Any] = {}
        # LLM_API_KEY carries the AWS access key id; secret comes from AWS_SECRET_ACCESS_KEY.
        if config.api_key:
            client_kwargs["aws_access_key_id"] = config.api_key
        region = (config.base_url or "").strip()
        if region:
            client_kwargs["region_name"] = region
        self._client = boto3.client("bedrock-runtime", **client_kwargs)
        self._model_id = _resolve_model_id(config.model)

    @measure_latency("BedrockLLM.acompletion")
    async def acompletion(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        import asyncio

        system, bedrock_msgs = _to_bedrock_messages(messages)
        params: Dict[str, Any] = {
            "modelId": self._model_id,
            "messages": bedrock_msgs,
            "inferenceConfig": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "maxTokens": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }
        if system:
            params["system"] = [{"text": system}]
        if self.config.top_p is not None:
            params["inferenceConfig"]["topP"] = self.config.top_p

        try:
            # boto3 is blocking; run in a thread so we don't block the event loop.
            resp = await asyncio.to_thread(self._client.converse, **params)
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "throttl" in msg:
                raise LLMRateLimitError(str(e)) from e
            if "timeout" in msg:
                raise LLMTimeoutError(str(e)) from e
            if "connect" in msg or "endpoint" in msg or "credential" in msg or "access" in msg:
                raise LLMConnectionError(str(e)) from e
            if "validation" in msg or ("model" in msg and "not" in msg):
                raise LLMConfigError(str(e)) from e
            raise LLMError(str(e)) from e

        text = ""
        try:
            for block in resp["output"]["message"]["content"]:
                if block.get("text"):
                    text += block["text"]
        except (KeyError, TypeError):
            text = ""
        stop = resp.get("stopReason")
        return {
            "content": text,
            "model": self._model_id,
            "finish_reason": stop,
            "raw": resp,
        }
