"""Abstract Base Classes (ABCs) defining structural domain interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime


class BaseEmbeddingProvider(ABC):
    """Abstract interface for vector embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate a floating point vector embedding for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of texts."""
        pass


class BaseLLMProvider(ABC):
    """Abstract interface for LLM completion providers."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute chat completion with optional tool calls."""
        pass
