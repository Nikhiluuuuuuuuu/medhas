"""LLM integration errors (provider-agnostic)."""


class LLMError(Exception):
    """Base error for all LLM-provider failures."""


class LLMConfigError(LLMError):
    """Invalid LLM configuration (missing key, unknown provider, bad model)."""


class LLMRateLimitError(LLMError):
    """Provider returned a 429 / quota-exceeded. Callers may back off and retry."""


class LLMTimeoutError(LLMError):
    """The provider did not respond within the configured timeout."""


class LLMConnectionError(LLMError):
    """Network-level failure talking to the provider."""
