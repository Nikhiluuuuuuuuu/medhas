"""Public surface for the :mod:`medhas.llm` subsystem."""
from .base import BaseLLM
from .config import LLMConfig
from .errors import (LLMError, LLMConfigError, LLMRateLimitError,
                     LLMTimeoutError, LLMConnectionError)
from .factory import create_llm, register_provider
from .router import LLMRouter, EXTRACTION, RESOLUTION, REASONING, SYNTHESIS
from .structured import generate_structured, generate_structured_default

__all__ = [
    "BaseLLM", "LLMConfig", "LLMError", "LLMConfigError", "LLMRateLimitError",
    "LLMTimeoutError", "LLMConnectionError", "create_llm", "register_provider",
    "LLMRouter", "EXTRACTION", "RESOLUTION", "REASONING", "SYNTHESIS",
    "generate_structured", "generate_structured_default",
]
