import pytest
import numpy as np
from medhas.storage.vector_index import VectorIndex

def test_vector_index_search():
    index = VectorIndex()
    vec1 = np.array([1.0] + [0.0]*383, dtype=np.float32)
    vec2 = np.array([0.0, 1.0] + [0.0]*382, dtype=np.float32)
    
    index.add_node("node1", vec1)
    index.add_node("node2", vec2)
    
    results = index.search(vec1, top_k=1)
    assert len(results) == 1
    assert results[0][0] == "node1"
    assert pytest.approx(results[0][1], 0.01) == 1.0
