"""LiteLLM provider — one interface to 100+ LLM APIs (OpenAI, Anthropic, Gemini,
Bedrock, Groq, Ollama, vLLM, Azure, DeepSeek, ...). Mirrors mem0's ``mem0/llms/litellm.py``
and cognee's ``litellm_instructor``: a single vendor-neutral path selected by ``model``
prefix (e.g. ``anthropic/claude-3-5-sonnet``, ``groq/llama-3.3-70b``).
"""
from typing import Any, Dict, List

from ..base import BaseLLM, Message
from ..config import LLMConfig
from ..errors import LLMRateLimitError, LLMTimeoutError, LLMConnectionError, LLMError
from ..metrics import measure_latency


class LiteLLMLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            import litellm
        except ImportError as e:  # pragma: no cover
            raise LLMError("`litellm` is required for LiteLLMLLM") from e
        self._litellm = litellm
        # litellm reads keys from env (OPENAI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY, ...)
        if config.api_key:
            self._litellm.api_key = config.api_key
        if config.base_url:
            self._litellm.api_base = config.base_url

    @measure_latency("LiteLLMLLM.acompletion")
    async def acompletion(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.base_url:
            params["api_base"] = self.config.base_url
        if "response_format" in kwargs:
            params["response_format"] = kwargs["response_format"]
        params.update(self.config.extra_params)
        try:
            resp = await self._litellm.acompletion(**params)
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                raise LLMRateLimitError(str(e)) from e
            if "timeout" in msg:
                raise LLMTimeoutError(str(e)) from e
            if "connect" in msg:
                raise LLMConnectionError(str(e)) from e
            raise LLMError(str(e)) from e
        content = resp.choices[0].message.content or ""
        return {"content": content, "model": self.config.model,
                "finish_reason": getattr(resp.choices[0], "finish_reason", None), "raw": resp}
