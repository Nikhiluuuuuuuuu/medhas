"""FastEmbed (ONNX) embedding provider — zero-network, sub-5ms local embeddings.

Kept as ``FastEmbeddingProvider`` for backward compatibility with the memory layer, while
the new provider-agnostic :class:`medhas.embeddings.BaseEmbedder` (``FastEmbedEmbedder``)
offers the same backend through the unified configuration interface.
"""
from typing import List, Optional

from medhas.core.interfaces import BaseEmbeddingProvider
from medhas.config import settings
from medhas.utils import logger, measure_latency
from medhas.core.exceptions import EmbeddingGenerationError

_GLOBAL_EMBEDDING_MODEL = None


class FastEmbeddingProvider(BaseEmbeddingProvider):
    """FastEmbed implementation generating sub-5ms vector embeddings."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name

    def _get_model(self):
        global _GLOBAL_EMBEDDING_MODEL
        if _GLOBAL_EMBEDDING_MODEL is None:
            try:
                from fastembed import TextEmbedding
                _GLOBAL_EMBEDDING_MODEL = TextEmbedding(model_name=self.model_name)
                logger.info(f"\u2713 Loaded FastEmbed model ONCE globally: {self.model_name}")
            except Exception as e:
                logger.error(f"FastEmbed model load error: {e}")
                raise EmbeddingGenerationError(f"FastEmbed initialization error: {e}")
        return _GLOBAL_EMBEDDING_MODEL

    async def embed_text(self, text: str) -> List[float]:
        async with measure_latency("FastEmbeddingProvider.embed_text"):
            try:
                model = self._get_model()
                embeddings = list(model.embed([text]))
                return embeddings[0].tolist()
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                raise EmbeddingGenerationError(f"Vector embedding failed: {e}")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        async with measure_latency(f"FastEmbeddingProvider.embed_batch ({len(texts)} items)"):
            try:
                model = self._get_model()
                embeddings = list(model.embed(texts))
                return [e.tolist() for e in embeddings]
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                raise EmbeddingGenerationError(f"Batch vector embedding failed: {e}")
