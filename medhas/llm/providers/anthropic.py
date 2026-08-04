"""Anthropic (Claude) provider via the Anthropic SDK."""
from typing import Any, Dict, List

from ..base import BaseLLM, Message
from ..config import LLMConfig
from ..errors import LLMRateLimitError, LLMTimeoutError, LLMConnectionError, LLMError
from ..metrics import measure_latency


class AnthropicLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover
            raise LLMError("`anthropic` is required for AnthropicLLM") from e
        self._client = AsyncAnthropic(api_key=config.api_key, base_url=config.base_url)

    @staticmethod
    def _split_system(messages: List[Message]) -> tuple[str, List[Dict[str, str]]]:
        system = ""
        convo = []
        for m in messages:
            if m.get("role") == "system":
                system += m["content"] + "\n"
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        return system.strip(), convo

    @measure_latency("AnthropicLLM.acompletion")
    async def acompletion(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        system, convo = self._split_system(messages)
        params = {
            "model": self.config.model,
            "messages": convo,
            "system": system,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        try:
            resp = await self._client.messages.create(**params)
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                raise LLMRateLimitError(str(e)) from e
            if "timeout" in msg:
                raise LLMTimeoutError(str(e)) from e
            if "connect" in msg:
                raise LLMConnectionError(str(e)) from e
            raise LLMError(str(e)) from e
        content = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return {"content": content, "model": resp.model, "finish_reason": resp.stop_reason, "raw": resp}
