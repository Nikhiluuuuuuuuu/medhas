import numpy as np
from typing import List, Tuple

class VectorIndex:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.node_ids: List[str] = []
        self.embeddings: List[np.ndarray] = []

    def add_node(self, node_id: str, embedding: List[float]):
        emb_arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_arr)
        if norm > 0:
            emb_arr = emb_arr / norm

        if node_id in self.node_ids:
            idx = self.node_ids.index(node_id)
            self.embeddings[idx] = emb_arr
        else:
            self.node_ids.append(node_id)
            self.embeddings.append(emb_arr)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.node_ids:
            return []

        q_arr = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        matrix = np.vstack(self.embeddings)
        sims = np.dot(matrix, q_arr)

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.node_ids[idx], float(sims[idx])) for idx in top_indices]
