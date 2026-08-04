"""Google Gemini provider via the google-generativeai SDK."""
from typing import Any, Dict, List

from ..base import BaseLLM, Message
from ..config import LLMConfig
from ..errors import LLMRateLimitError, LLMTimeoutError, LLMConnectionError, LLMError
from ..metrics import measure_latency


class GeminiLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            import google.generativeai as genai
        except ImportError as e:  # pragma: no cover
            raise LLMError("`google-generativeai` is required for GeminiLLM") from e
        genai.configure(api_key=config.api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(config.model)

    @measure_latency("GeminiLLM.acompletion")
    async def acompletion(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        history = [{"role": ("model" if m["role"] == "assistant" else "user"), "parts": [m["content"]]}
                   for m in messages if m.get("role") != "system"]
        prompt = "\n".join(sys_parts)
        if history:
            last = history[-1]
            prompt = (prompt + "\n" + last["parts"][0]) if prompt else last["parts"][0]
            history = history[:-1]
        try:
            resp = await self._model.generate_content_async(
                prompt,
                generation_config={"temperature": kwargs.get("temperature", self.config.temperature),
                                   "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens)},
            )
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                raise LLMRateLimitError(str(e)) from e
            if "timeout" in msg:
                raise LLMTimeoutError(str(e)) from e
            if "connect" in msg:
                raise LLMConnectionError(str(e)) from e
            raise LLMError(str(e)) from e
        return {"content": resp.text, "model": self.config.model, "finish_reason": None, "raw": resp}
