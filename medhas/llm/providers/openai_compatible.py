"""OpenAI-compatible provider (OpenAI, Groq, Together, Ollama-v1, vLLM, LM Studio, ...).

This single class covers every vendor that speaks the OpenAI Chat Completions HTTP API.
Groq is reachable here with ``base_url="https://api.groq.com/openai/v1"`` — there is NO
dedicated Groq code anywhere in the engine, exactly as the reference engines treat any
OpenAI-compatible endpoint as a URL swap.
"""
from typing import Any, Dict, List

from ..base import BaseLLM, Message
from ..config import LLMConfig
from ..errors import LLMRateLimitError, LLMTimeoutError, LLMConnectionError, LLMError
from ..metrics import measure_latency


class OpenAICompatibleLLM(BaseLLM):
    """Talks to any OpenAI-style /v1/chat/completions endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise LLMError("`openai` is required for OpenAICompatibleLLM") from e
        self._client = AsyncOpenAI(
            api_key=config.api_key or "not-needed",
            base_url=config.base_url,
            timeout=config.timeout,
        )

    @measure_latency("OpenAICompatibleLLM.acompletion")
    async def acompletion(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if self.config.top_p is not None and "top_p" not in kwargs:
            params["top_p"] = self.config.top_p
        if "response_format" in kwargs:
            params["response_format"] = kwargs["response_format"]
        params.update(self.config.extra_params)
        params.update({k: v for k, v in kwargs.items()
                       if k not in ("temperature", "max_tokens", "response_format", "top_p")})
        try:
            resp = await self._client.chat.completions.create(**params)
        except Exception as e:  # normalise provider-agnostic errors
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "quota" in msg:
                raise LLMRateLimitError(str(e)) from e
            if "timeout" in msg:
                raise LLMTimeoutError(str(e)) from e
            if "connect" in msg or "connection" in msg:
                raise LLMConnectionError(str(e)) from e
            raise LLMError(str(e)) from e
        content = resp.choices[0].message.content or ""
        return {
            "content": content,
            "model": getattr(resp, "model", self.config.model),
            "finish_reason": getattr(resp.choices[0], "finish_reason", None),
            "raw": resp,
        }
