"""Groq API client wrapper with active models hierarchy and zero mock data."""

import asyncio
import json
from typing import List, Dict, Any, Optional
from groq import AsyncGroq, RateLimitError
from core.interfaces import BaseLLMProvider
from config import settings
from utils import logger, measure_latency
from core.exceptions import LLMProviderError

# Currently active non-decommissioned Groq model hierarchy
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-specdec"
]


class GroqLLMProvider(BaseLLMProvider):
    """Production Groq LLM client wrapper with multi-model fallback and exponential backoff."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self._client: Optional[AsyncGroq] = None

    def _get_client(self) -> AsyncGroq:
        if self._client is None:
            if not self.api_key:
                raise LLMProviderError(
                    "GROQ_API_KEY is missing from environment. Cannot proceed without API key.")
            self._client = AsyncGroq(api_key=self.api_key)
        return self._client

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute async chat completion with automatic model fallback on 429 rate limit."""
        async with measure_latency(f"GroqLLMProvider.chat_completion ({self.model})"):
            client = self._get_client()
            candidate_models = [self.model] + \
                [m for m in FALLBACK_MODELS if m != self.model]

            last_error = None
            for model_name in candidate_models:
                for attempt in range(3):
                    try:
                        kwargs: Dict[str, Any] = {
                            "model": model_name,
                            "messages": messages,
                            "temperature": temperature if temperature is not None else settings.GROQ_TEMPERATURE,
                            "max_tokens": settings.GROQ_MAX_TOKENS,
                        }
                        if tools:
                            kwargs["tools"] = tools
                            kwargs["tool_choice"] = "auto"

                        response = await client.chat.completions.create(**kwargs)
                        choice = response.choices[0]
                        message = choice.message

                        result: Dict[str, Any] = {
                            "content": message.content or "",
                            "role": message.role,
                            "tool_calls": []
                        }

                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tc in message.tool_calls:
                                result["tool_calls"].append({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    }
                                })
                        return result

                    except RateLimitError as rle:
                        last_error = rle
                        sleep_time = 2.0 * (attempt + 1)
                        logger.warning(
                            f"⚠️ Rate limit (429) on model '{model_name}'. Waiting {sleep_time}s before retrying...")
                        await asyncio.sleep(sleep_time)
                        continue

                    except Exception as e:
                        last_error = e
                        logger.warning(
                            f"⚠️ Groq API model notice on '{model_name}': {e}. Switching candidate model...")
                        break

            raise LLMProviderError(
                f"All Groq LLM candidate models exhausted or rate-limited. Last error: {last_error}")
