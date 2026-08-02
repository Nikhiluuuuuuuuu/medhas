"""Ultra-low latency vector embedding provider using FastEmbed (ONNX) with zero mock fallbacks."""

from typing import List, Optional
from core.interfaces import BaseEmbeddingProvider
from config import settings
from utils import logger, measure_latency
from core.exceptions import EmbeddingGenerationError

_GLOBAL_EMBEDDING_MODEL = None

class FastEmbeddingProvider(BaseEmbeddingProvider):
    """FastEmbed implementation generating sub-5ms vector embeddings."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name

    def _get_model(self):
        """Lazy load FastEmbed TextEmbedding model once globally."""
        global _GLOBAL_EMBEDDING_MODEL
        if _GLOBAL_EMBEDDING_MODEL is None:
            try:
                from fastembed import TextEmbedding
                _GLOBAL_EMBEDDING_MODEL = TextEmbedding(model_name=self.model_name)
                logger.info(f"✅ Loaded FastEmbed model ONCE globally: {self.model_name}")
            except Exception as e:
                logger.error(f"FastEmbed model load error: {e}")
                raise EmbeddingGenerationError(f"FastEmbed initialization error: {e}")
        return _GLOBAL_EMBEDDING_MODEL

    async def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for a single string."""
        async with measure_latency("FastEmbeddingProvider.embed_text"):
            try:
                model = self._get_model()
                embeddings = list(model.embed([text]))
                return embeddings[0].tolist()
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                raise EmbeddingGenerationError(f"Vector embedding failed: {e}")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of strings."""
        async with measure_latency(f"FastEmbeddingProvider.embed_batch ({len(texts)} items)"):
            try:
                model = self._get_model()
                embeddings = list(model.embed(texts))
                return [e.tolist() for e in embeddings]
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                raise EmbeddingGenerationError(f"Batch vector embedding failed: {e}")
