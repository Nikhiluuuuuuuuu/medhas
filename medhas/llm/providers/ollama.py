"""Ollama provider (local models) — uses the OpenAI-compatible Ollama endpoint."""
from typing import Any, Dict, List

from .openai_compatible import OpenAICompatibleLLM


class OllamaLLM(OpenAICompatibleLLM):
    """Ollama exposes an OpenAI-compatible API at http://host:11434/v1 by default."""

    def __init__(self, config):
        # Ensure a sane default base_url for local Ollama if none supplied.
        if not config.base_url:
            config = config.merged(base_url="http://localhost:11434/v1")
        super().__init__(config)
