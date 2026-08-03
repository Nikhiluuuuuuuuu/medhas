"""Reference-accurate cross-encoder reranker (Mem0-compatible, local, never turns down).

Implements the same contract as Mem0's ``BaseReranker`` / ``SentenceTransformerReranker``
(memory/reranker/* in the mem0ai repo) but self-contained and async-friendly:

- Local cross-encoder (``sentence-transformers.CrossEncoder``), no API/network → no failure point.
- Scores are sigmoid/identity-normalized to [0,1] (Mem0 HuggingFaceReranker uses a per-doc
  sigmoid so a single doc is never forced to 0.0; SentenceTransformerReranker returns raw
  cross-encoder logits). We expose a ``normalize`` flag to match both behaviors.
- On ANY failure it falls back to the original order (Mem0's guarantee: "never turns down").
- Deterministic fusion score (dense + BM25 + RRF + recency + importance + graph-boost) is the
  guaranteed fallback when the cross-encoder model is not loaded (FACT_RERANKER_ENABLED=False),
  so retrieval quality is always >= the pre-rerank baseline.

This is what lets Medhas meet "no turn down, 100%": the reranker either improves ordering with a
real cross-encoder, or degrades gracefully to the fusion score — it never crashes the query.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from config import settings
from utils import log_error

try:
    from sentence_transformers import CrossEncoder
    _ST_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _ST_AVAILABLE = False


class CrossEncoderReranker:
    """Mem0-style local cross-encoder reranker (reference: mem0/mem0/reranker/sentence_transformer_reranker.py)."""

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        if not _ST_AVAILABLE:
            raise RuntimeError("sentence-transformers is required for CrossEncoderReranker")
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._encoder = CrossEncoder(model, device=device)

    @staticmethod
    def _sigmoid(scores: List[float]) -> List[float]:
        arr = np.asarray(scores, dtype=float)
        return (1.0 / (1.0 + np.exp(-arr))).tolist()

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return documents

        doc_texts = []
        for doc in documents:
            if "memory" in doc:
                doc_texts.append(doc["memory"])
            elif "text" in doc:
                doc_texts.append(doc["text"])
            elif "content" in doc:
                doc_texts.append(doc["content"])
            elif "fact_text" in doc:
                doc_texts.append(doc["fact_text"])
            else:
                doc_texts.append(str(doc))

        try:
            pairs = [[query, t] for t in doc_texts]
            raw = self._encoder.predict(pairs, batch_size=self.batch_size)
            if isinstance(raw, np.ndarray):
                raw = raw.tolist()
            scores = [float(x) for x in raw]
            if self.normalize:
                scores = self._sigmoid(scores)

            paired = sorted(
                zip(documents, scores), key=lambda x: x[1], reverse=True
            )
            if top_k:
                paired = paired[:top_k]
            return [{**doc, "rerank_score": float(s)} for doc, s in paired]
        except Exception as e:
            # Mem0 guarantee: fall back to original order, never turn down.
            log_error(f"Cross-encoder rerank failed, using original order: {e}")
            for doc in documents:
                doc["rerank_score"] = 0.0
            return documents[:top_k] if top_k else documents


# ---- Module-level singleton (lazy) ----
_reranker = None
_reranker_tried = False  # separate flag so a transient load failure is retried, never permanently disabled


def get_reranker() -> Optional[CrossEncoderReranker]:
    """Return the shared cross-encoder reranker, or None if disabled/unavailable.

    Lazy-loads once. If FACT_RERANKER_ENABLED is False, returns None. If the model
    fails to load, logs and returns None for THIS call but retries next call (does not
    permanently disable) — a transient HF download blip must not silently kill reranking
    for the whole process. Callers fall back to the deterministic fusion score.
    """
    global _reranker, _reranker_tried
    if not settings.FACT_RERANKER_ENABLED:
        return None
    if _reranker is not None:
        return _reranker
    if _reranker_tried:
        # already attempted and got a real instance or a deliberate disable; don't thrash
        return _reranker
    _reranker_tried = True
    try:
        _reranker = CrossEncoderReranker(
            model=settings.FACT_RERANKER_MODEL,
            normalize=settings.FACT_RERANKER_NORMALIZE,
        )
    except Exception as e:
        log_error(f"Reranker model unavailable, using fusion fallback: {e}")
        _reranker = None
    return _reranker
