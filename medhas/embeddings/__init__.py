"""Provider-agnostic embedding integration (cognee ``LiteLLMEmbeddingEngine`` style).

The engine never imports a specific embedding vendor. A single ``EmbedderConfig`` selects
the backend; ``create_embedder`` builds it. Local ONNX (fastembed) remains the default
zero-network option, while OpenAI-compatible / LiteLLM / Ollama backends are swappable.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .embedding_provider import FastEmbeddingProvider


class EmbedderConfig(BaseModel):
    provider: str = "fastembed"          # fastembed | openai | ollama | litellm
    model: str = "BAAI/bge-base-en-v1.5"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimensions: Optional[int] = None
    extra_params: dict = Field(default_factory=dict)


class BaseEmbedder(ABC):
    @abstractmethod
    async def aembed(self, texts: List[str]) -> List[List[float]]: ...

    async def aembed_one(self, text: str) -> List[float]:
        return (await self.aembed([text]))[0]


class FastEmbedEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=config.model)

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        import asyncio
        return await asyncio.to_thread(lambda: list(self._model.embed(texts)))


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=config.api_key or "not-needed", base_url=config.base_url)

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        resp = await self._client.embeddings.create(model=self.config.model, input=texts)
        return [d.embedding for d in resp.data]


def create_embedder(config: EmbedderConfig) -> BaseEmbedder:
    mapping = {
        "fastembed": FastEmbedEmbedder,
        "openai": OpenAIEmbedder,
        "ollama": OpenAIEmbedder,
        "litellm": OpenAIEmbedder,
    }
    cls = mapping.get(config.provider.lower())
    if cls is None:
        raise ValueError(f"Unknown embedder provider: {config.provider!r}. Known: {sorted(mapping)}")
    return cls(config)


__all__ = [
    "EmbedderConfig", "BaseEmbedder", "FastEmbedEmbedder", "OpenAIEmbedder",
    "create_embedder", "FastEmbeddingProvider",
]
