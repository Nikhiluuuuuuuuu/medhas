import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class VectorIndex:
    """
    Multilingual Dense Vector Storage and Cosine Similarity Index.
    Stores dense embeddings for both entity nodes and raw text / fact triple chunks.
    Supports dynamic vector dimension resizing.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.node_ids: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_node(self, node_id: str, embedding: Any, meta: Optional[Dict[str, Any]] = None):
        if embedding is None or len(embedding) == 0:
            return
        emb_arr = np.array(embedding, dtype=np.float32)

        if self.dim != len(emb_arr):
            self.dim = len(emb_arr)

        norm = np.linalg.norm(emb_arr)
        if norm > 0:
            emb_arr = emb_arr / norm

        meta = meta or {"type": "entity"}

        if node_id in self.node_ids:
            idx = self.node_ids.index(node_id)
            self.embeddings[idx] = emb_arr
            self.metadata[idx] = meta
        else:
            self.node_ids.append(node_id)
            self.embeddings.append(emb_arr)
            self.metadata.append(meta)

    def search(self, query_embedding: Any, top_k: int = 5, item_type: Optional[str] = None) -> List[Tuple[str, float]]:
        if not self.node_ids or query_embedding is None or len(query_embedding) == 0:
            return []


        q_arr = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        if item_type is not None:
            indices = [i for i, m in enumerate(self.metadata) if m.get("type") == item_type]
            if not indices:
                return []
            filtered_embs = [self.embeddings[i] for i in indices]
            filtered_ids = [self.node_ids[i] for i in indices]
            matrix = np.vstack(filtered_embs)
            sims = np.dot(matrix, q_arr)
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [(filtered_ids[idx], float(sims[idx])) for idx in top_indices]

        matrix = np.vstack(self.embeddings)
        sims = np.dot(matrix, q_arr)

        top_indices = np.argsort(sims)[::-1][:min(top_k, len(self.node_ids))]
        return [(self.node_ids[idx], float(sims[idx])) for idx in top_indices]

