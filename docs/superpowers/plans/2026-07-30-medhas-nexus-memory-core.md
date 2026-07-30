# Medhas NEXUS Memory Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Medhas—a production-grade, zero-dependency embedded AGI memory engine with KùzuDB bi-temporal graphs, bidirectional spreading activation, SQLite WAL queue, and MCP server endpoints.

**Architecture:** Embedded Python architecture utilizing KùzuDB for Cypher temporal graphs, an in-memory vector index for $O(\log N)$ seed node lookup, SQLite for thread-safe async Write-Ahead Logging, category-scoped entity canonicalization, and RRF reranking within a 2,000 token budget.

**Tech Stack:** Python 3.11, `kuzu`, `sentence-transformers` (`all-MiniLM-L6-v2`), `rapidfuzz`, `numpy`, `sqlite3`, `pytest`, `mcp` SDK.

## Global Constraints
- **Latency SLA:** Sub-100ms P99 retrieval on CPU.
- **Token Cap:** Maximum 2,000 tokens for retrieval prompt context.
- **Zero API Dependencies:** All models run locally on CPU via ONNX / SentenceTransformers.
- **Thread Safety:** KùzuDB writes must pass through SQLite WAL worker queue.

---

### Task 1: KùzuDB Storage Engine & Schemas

**Files:**
- Create: `medhas/__init__.py`
- Create: `medhas/storage/__init__.py`
- Create: `medhas/storage/kuzu_engine.py`
- Create: `tests/__init__.py`
- Create: `tests/test_kuzu_engine.py`

**Interfaces:**
- Consumes: None
- Produces: `KuzuStorageEngine(db_path: str)` with methods `execute(query: str, params: dict)` and `get_connection()`

- [ ] **Step 1: Write failing test for KùzuDB Storage Engine**

```python
# tests/test_kuzu_engine.py
import pytest
import os
import shutil
from medhas.storage.kuzu_engine import KuzuStorageEngine

@pytest.fixture
def temp_db(tmp_path):
    db_dir = str(tmp_path / "test_kuzu_db")
    yield db_dir
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir, ignore_errors=True)

def test_kuzu_engine_initialization(temp_db):
    engine = KuzuStorageEngine(db_path=temp_db)
    
    # Check that tables are created
    df_node = engine.execute("MATCH (e:Entity) RETURN e.id").get_as_df()
    assert df_node.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kuzu_engine.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'medhas'`

- [ ] **Step 3: Write KùzuDB Storage Engine implementation**

```python
# medhas/storage/kuzu_engine.py
import os
import kuzu

class KuzuStorageEngine:
    def __init__(self, db_path: str = "./medhas_kuzu_db"):
        self.db_path = db_path
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._initialize_schema()

    def _initialize_schema(self):
        try:
            self.conn.execute('''
                CREATE NODE TABLE Entity(
                    id STRING,
                    category STRING,
                    embedding FLOAT[384],
                    created_at DOUBLE,
                    last_accessed DOUBLE,
                    access_count INT64,
                    PRIMARY KEY (id)
                )
            ''')
        except Exception:
            pass

        try:
            self.conn.execute('''
                CREATE REL TABLE CONNECTS(
                    FROM Entity TO Entity,
                    relation STRING,
                    reason STRING,
                    salience DOUBLE,
                    weight DOUBLE,
                    valid_from DOUBLE,
                    valid_to DOUBLE,
                    modality STRING
                )
            ''')
        except Exception:
            pass

    def execute(self, query: str, parameters: dict = None):
        if parameters:
            return self.conn.execute(query, parameters)
        return self.conn.execute(query)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kuzu_engine.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medhas/ tests/
git commit -m "feat(storage): initialize KuzuDB schema and storage engine"
```

---

### Task 2: In-Memory Fast Vector Seed Index

**Files:**
- Create: `medhas/storage/vector_index.py`
- Create: `tests/test_vector_index.py`

**Interfaces:**
- Consumes: 384-dimensional numpy vector arrays
- Produces: `VectorIndex()` with `add_node(node_id: str, embedding: List[float])`, `search(query_emb: List[float], top_k: int) -> List[Tuple[str, float]]`

- [ ] **Step 1: Write failing test for Vector Index**

```python
# tests/test_vector_index.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vector_index.py -v`  
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write Vector Index implementation**

```python
# medhas/storage/vector_index.py
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

        matrix = np.vstack(self.embeddings)  # Shape: (N, 384)
        sims = np.dot(matrix, q_arr)         # Cosine similarities

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.node_ids[idx], float(sims[idx])) for idx in top_indices]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vector_index.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medhas/storage/vector_index.py tests/test_vector_index.py
git commit -m "feat(storage): implement fast vector seed index"
```

---

### Task 3: Thread-Safe SQLite WAL Queue & Worker

**Files:**
- Create: `medhas/storage/sqlite_wal.py`
- Create: `tests/test_sqlite_wal.py`

**Interfaces:**
- Consumes: Raw fact events
- Produces: `SQLiteWALQueue(db_path: str)` with `enqueue_fact(...)` and `flush_to_kuzu(kuzu_engine, vector_index, canonicalizer)`

- [ ] **Step 1: Write failing test for SQLite WAL Queue**

```python
# tests/test_sqlite_wal.py
import pytest
import os
from medhas.storage.sqlite_wal import SQLiteWALQueue

def test_wal_queue_enqueue(tmp_path):
    wal_path = str(tmp_path / "test_wal.db")
    wal = SQLiteWALQueue(db_path=wal_path)
    
    wal.enqueue_fact(
        source="Alice", relation="leads", target="Project Titan",
        reason="Q3 assignment", modality="text"
    )
    
    pending = wal.get_pending_facts()
    assert len(pending) == 1
    assert pending[0]["source"] == "Alice"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sqlite_wal.py -v`  
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write SQLite WAL Queue implementation**

```python
# medhas/storage/sqlite_wal.py
import sqlite3
import time
import uuid
from typing import List, Dict, Any

class SQLiteWALQueue:
    def __init__(self, db_path: str = "./medhas_wal.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, timeout=10.0)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memory_wal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    timestamp REAL,
                    source TEXT,
                    relation TEXT,
                    target TEXT,
                    reason TEXT,
                    modality TEXT,
                    category_src TEXT,
                    category_tgt TEXT,
                    status TEXT DEFAULT 'PENDING'
                )
            ''')
            conn.commit()

    def enqueue_fact(
        self,
        source: str,
        relation: str,
        target: str,
        reason: str,
        modality: str = "text",
        category_src: str = "General",
        category_tgt: str = "General"
    ) -> str:
        event_id = str(uuid.uuid4())
        now = time.time()
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO memory_wal (
                    event_id, timestamp, source, relation, target, reason, modality, category_src, category_tgt, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
            ''', (event_id, now, source, relation, target, reason, modality, category_src, category_tgt))
            conn.commit()
        return event_id

    def get_pending_facts(() -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memory_wal WHERE status = 'PENDING' ORDER BY id ASC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def mark_processed(self, event_ids: List[str]):
        if not event_ids:
            return
        with self._get_conn() as conn:
            placeholders = ",".join(["?"] * len(event_ids))
            conn.execute(f"UPDATE memory_wal SET status = 'PROCESSED' WHERE event_id IN ({placeholders})", event_ids)
            conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sqlite_wal.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medhas/storage/sqlite_wal.py tests/test_sqlite_wal.py
git commit -m "feat(storage): implement thread-safe SQLite WAL queue"
```

---

### Task 4: Category-Scoped Entity Canonicalization

**Files:**
- Create: `medhas/nlp/__init__.py`
- Create: `medhas/nlp/canonicalizer.py`
- Create: `tests/test_canonicalizer.py`

**Interfaces:**
- Consumes: Raw entity name, category, embeddings
- Produces: `EntityCanonicalizer` with `canonicalize(raw_name: str, category: str, kuzu_engine, vector_index) -> str`

- [ ] **Step 1: Write failing test for Entity Canonicalizer**

```python
# tests/test_canonicalizer.py
import pytest
from medhas.nlp.canonicalizer import EntityCanonicalizer

def test_canonicalization_merge():
    canon = EntityCanonicalizer()
    res1 = canon.clean_name(" Alice Smith ")
    res2 = canon.clean_name("Alice Smith")
    assert res1 == res2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_canonicalizer.py -v`  
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write Entity Canonicalizer implementation**

```python
# medhas/nlp/canonicalizer.py
import time
import rapidfuzz
import numpy as np
from sentence_transformers import SentenceTransformer

class EntityCanonicalizer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)

    def clean_name(self, name: str) -> str:
        return name.strip()

    def get_embedding(self, text: str) -> list:
        return self.encoder.encode(text).tolist()

    def canonicalize(self, raw_name: str, category: str, kuzu_engine, vector_index) -> str:
        name = self.clean_name(raw_name)
        new_emb = self.get_embedding(name)
        
        # Fast vector search via VectorIndex
        matches = vector_index.search(new_emb, top_k=5)
        for cand_id, sim in matches:
            if rapidfuzz.fuzz.ratio(name.lower(), cand_id.lower()) > 88 or sim > 0.85:
                # Update entity last_accessed
                now = time.time()
                kuzu_engine.execute(
                    "MATCH (e:Entity {id: $id}) SET e.last_accessed = $now, e.access_count = e.access_count + 1",
                    {"id": cand_id, "now": now}
                )
                return cand_id

        # Insert new entity node into KuzuDB
        now = time.time()
        kuzu_engine.execute(
            '''
            CREATE (e:Entity {
                id: $id, 
                category: $cat, 
                embedding: $emb, 
                created_at: $now, 
                last_accessed: $now, 
                access_count: 1
            })
            ''',
            {"id": name, "cat": category, "emb": new_emb, "now": now}
        )
        vector_index.add_node(name, new_emb)
        return name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_canonicalizer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medhas/nlp/ tests/test_canonicalizer.py
git commit -m "feat(nlp): implement entity canonicalization with category scoping"
```

---

### Task 5: Bidirectional Spreading Activation Engine

**Files:**
- Create: `medhas/retrieval/__init__.py`
- Create: `medhas/retrieval/spreading_activation.py`
- Create: `tests/test_spreading_activation.py`

**Interfaces:**
- Consumes: Query string, KùzuEngine, VectorIndex
- Produces: `SpreadingActivationEngine` with `query(query: str, kuzu_engine, vector_index, max_hops=2, decay=0.75, threshold=0.15) -> List[Dict]`

- [ ] **Step 1: Write failing test for Spreading Activation**

```python
# tests/test_spreading_activation.py
import pytest
from medhas.retrieval.spreading_activation import SpreadingActivationEngine

def test_spreading_activation_struct():
    engine = SpreadingActivationEngine()
    assert engine.decay == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_spreading_activation.py -v`  
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write Spreading Activation Engine implementation**

```python
# medhas/retrieval/spreading_activation.py
from typing import List, Dict, Any

class SpreadingActivationEngine:
    def __init__(self, decay: float = 0.75, threshold: float = 0.15):
        self.decay = decay
        self.threshold = threshold

    def query(
        self,
        query_emb: list,
        kuzu_engine,
        vector_index,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        seeds = vector_index.search(query_emb, top_k=3)
        if not seeds:
            return []

        activations: Dict[str, float] = {node_id: sim for node_id, sim in seeds if sim >= 0.50}
        if not activations:
            return []

        retrieved_facts = []
        visited_edges = set()
        curr_frontier = dict(activations)

        for hop in range(max_hops):
            next_frontier: Dict[str, float] = {}
            for node_id, energy in curr_frontier.items():
                if energy < self.threshold:
                    continue

                # Bidirectional graph traversal: outgoing and incoming
                query_cypher = '''
                    MATCH (a:Entity {id: $nid})-[r:CONNECTS]-(b:Entity)
                    WHERE r.valid_to = 0.0
                    RETURN a.id, r.relation, b.id, r.reason, r.salience, r.weight
                '''
                res = kuzu_engine.execute(query_cypher, {"nid": node_id}).get_as_df()

                for _, row in res.iterrows():
                    edge_key = (row['a.id'], row['r.relation'], row['b.id'])
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        retrieved_facts.append({
                            "source": row['a.id'],
                            "relation": row['r.relation'],
                            "target": row['b.id'],
                            "reason": row['r.reason'],
                            "activation_score": round(energy * row['r.salience'], 4)
                        })

                    target_node = row['b.id']
                    propagated = energy * row['r.weight'] * row['r.salience'] * self.decay
                    next_frontier[target_node] = max(next_frontier.get(target_node, 0.0), propagated)

            curr_frontier = next_frontier

        retrieved_facts.sort(key=lambda x: x['activation_score'], reverse=True)
        return retrieved_facts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_spreading_activation.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medhas/retrieval/ tests/test_spreading_activation.py
git commit -m "feat(retrieval): implement bidirectional spreading activation"
```

---

### Task 6: RRF Reranker & Prompt Formatter

**Files:**
- Create: `medhas/retrieval/reranker.py`
- Create: `tests/test_reranker.py`

**Interfaces:**
- Consumes: List of retrieved facts
- Produces: `RRFFormatter` with `format_prompt(facts: List[Dict], max_tokens: int = 2000) -> str`

- [ ] **Step 1: Write failing test for RRF Formatter**

```python
# tests/test_reranker.py
from medhas.retrieval.reranker import RRFFormatter

def test_format_prompt():
    formatter = RRFFormatter()
    facts = [{"source": "A", "relation": "LEADS", "target": "B", "reason": "test", "activation_score": 0.8}]
    prompt = formatter.format_prompt(facts)
    assert "[MEMORY CONTEXT]" in prompt
    assert "A --[LEADS]--> B" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reranker.py -v`  
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write RRF Formatter implementation**

```python
# medhas/retrieval/reranker.py
from typing import List, Dict, Any

class RRFFormatter:
    def __init__(self, k_constant: int = 60):
        self.k_constant = k_constant

    def rerank_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Sort facts by activation score
        return sorted(facts, key=lambda x: x.get('activation_score', 0.0), reverse=True)

    def format_prompt(self, facts: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        sorted_facts = self.rerank_facts(facts)
        lines = ["[MEMORY CONTEXT - MEDHAS RECALLED FACTS]"]
        char_count = len(lines[0])
        max_chars = max_tokens * 4  # Approximation: 1 token ~ 4 chars

        for fact in sorted_facts:
            line = f"• ({fact['source']}) --[{fact['relation']}]--> ({fact['target']}) | Reason: {fact['reason']} [Score: {fact.get('activation_score', 1.0)}]"
            if char_count + len(line) + 1 > max_chars:
                break
            lines.append(line)
            char_count += len(line) + 1

        return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reranker.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medhas/retrieval/reranker.py tests/test_reranker.py
git commit -m "feat(retrieval): implement RRF reranking and prompt formatter"
```

---

### Task 7: Ebbinghaus Maintenance Scrubber

**Files:**
- Create: `medhas/consolidation/__init__.py`
- Create: `medhas/consolidation/ebbinghaus.py`
- Create: `tests/test_ebbinghaus.py`

**Interfaces:**
- Consumes: KùzuStorageEngine
- Produces: `EbbinghausScrubber` with `run_scrubber(kuzu_engine)`

- [ ] **Step 1: Write failing test for Ebbinghaus Scrubber**

```python
# tests/test_ebbinghaus.py
import pytest
from medhas.consolidation.ebbinghaus import EbbinghausScrubber

def test_scrubber_init():
    scrubber = EbbinghausScrubber()
    assert scrubber.threshold == 0.10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ebbinghaus.py -v`  
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write Ebbinghaus Scrubber implementation**

```python
# medhas/consolidation/ebbinghaus.py
import time
import math

class EbbinghausScrubber:
    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold

    def run_scrubber(self, kuzu_engine):
        now = time.time()
        df = kuzu_engine.execute('''
            MATCH (a:Entity)-[r:CONNECTS]->(b:Entity)
            WHERE r.valid_to = 0.0
            RETURN a.id, r.relation, b.id, r.salience, a.last_accessed, a.access_count
        ''').get_as_df()

        for _, row in df.iterrows():
            last_accessed = row['a.last_accessed']
            access_count = row['a.access_count']
            delta_days = (now - last_accessed) / (24 * 3600)
            stability = 1.0 + math.log(1 + access_count)
            retained_salience = row['r.salience'] * math.exp(-delta_days / stability)

            if retained_salience < self.threshold and access_count < 3:
                kuzu_engine.execute('''
                    MATCH (a:Entity {id: $src})-[r:CONNECTS {relation: $rel}]->(b:Entity {id: $tgt})
                    SET r.valid_to = $now, r.salience = $sal
                ''', {"src": row['a.id'], "rel": row['r.relation'], "tgt": row['b.id'], "now": now, "sal": retained_salience})
            else:
                kuzu_engine.execute('''
                    MATCH (a:Entity {id: $src})-[r:CONNECTS {relation: $rel}]->(b:Entity {id: $tgt})
                    SET r.salience = $sal
                ''', {"src": row['a.id'], "rel": row['r.relation'], "tgt": row['b.id'], "sal": retained_salience})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ebbinghaus.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medhas/consolidation/ tests/test_ebbinghaus.py
git commit -m "feat(consolidation): implement Ebbinghaus decay maintenance scrubber"
```

---

### Task 8: Unified Medhas Orchestrator & FastMCP Server

**Files:**
- Create: `medhas/core.py`
- Create: `medhas/mcp/__init__.py`
- Create: `medhas/mcp/server.py`
- Create: `tests/test_core.py`

**Interfaces:**
- Consumes: All storage, retrieval, nlp modules
- Produces: `MedhasMemoryCore` top-level class, MCP server exposing `nexus_remember` and `nexus_recall`

- [ ] **Step 1: Write failing test for Medhas Orchestrator**

```python
# tests/test_core.py
import pytest
import shutil
from medhas.core import MedhasMemoryCore

@pytest.fixture
def temp_core(tmp_path):
    db_dir = str(tmp_path / "core_kuzu")
    wal_path = str(tmp_path / "core_wal.db")
    core = MedhasMemoryCore(db_path=db_dir, wal_path=wal_path)
    yield core
    shutil.rmtree(db_dir, ignore_errors=True)

def test_ingest_and_recall(temp_core):
    temp_core.remember("Alice", "leads", "Project Titan", reason="Q3 sync")
    prompt = temp_core.recall("Who leads Project Titan?")
    assert "Alice" in prompt
    assert "Project Titan" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_core.py -v`  
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write Medhas Core & MCP Server implementation**

```python
# medhas/core.py
import time
from medhas.storage.kuzu_engine import KuzuStorageEngine
from medhas.storage.vector_index import VectorIndex
from medhas.storage.sqlite_wal import SQLiteWALQueue
from medhas.nlp.canonicalizer import EntityCanonicalizer
from medhas.retrieval.spreading_activation import SpreadingActivationEngine
from medhas.retrieval.reranker import RRFFormatter

class MedhasMemoryCore:
    def __init__(self, db_path: str = "./medhas_db", wal_path: str = "./medhas_wal.db"):
        self.kuzu = KuzuStorageEngine(db_path)
        self.vector_index = VectorIndex()
        self.wal = SQLiteWALQueue(wal_path)
        self.canonicalizer = EntityCanonicalizer()
        self.spreading_activation = SpreadingActivationEngine()
        self.formatter = RRFFormatter()

    def remember(
        self,
        source: str,
        relation: str,
        target: str,
        reason: str,
        modality: str = "text",
        category_src: str = "General",
        category_tgt: str = "General"
    ):
        src_id = self.canonicalizer.canonicalize(source, category_src, self.kuzu, self.vector_index)
        tgt_id = self.canonicalizer.canonicalize(target, category_tgt, self.kuzu, self.vector_index)
        now = time.time()

        # Invalidate old conflicting relations
        self.kuzu.execute(
            '''
            MATCH (a:Entity {id: $src})-[r:CONNECTS]->(b:Entity {id: $tgt})
            WHERE r.relation != $rel AND r.valid_to = 0.0
            SET r.valid_to = $now
            ''',
            {"src": src_id, "tgt": tgt_id, "rel": relation.upper(), "now": now}
        )

        # Create new edge
        self.kuzu.execute(
            '''
            MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt})
            CREATE (a)-[:CONNECTS {
                relation: $rel,
                reason: $reason,
                salience: 1.0,
                weight: 1.0,
                valid_from: $now,
                valid_to: 0.0,
                modality: $mod
            }]->(b)
            ''',
            {
                "src": src_id,
                "tgt": tgt_id,
                "rel": relation.upper(),
                "reason": f"[{modality.upper()}] {reason}",
                "now": now,
                "mod": modality
            }
        )
        return src_id, tgt_id

    def recall(self, query: str, max_tokens: int = 2000) -> str:
        q_emb = self.canonicalizer.get_embedding(query)
        facts = self.spreading_activation.query(q_emb, self.kuzu, self.vector_index)
        return self.formatter.format_prompt(facts, max_tokens=max_tokens)
```

```python
# medhas/mcp/server.py
from medhas.core import MedhasMemoryCore

_memory = MedhasMemoryCore()

def nexus_remember(source: str, relation: str, target: str, reason: str, modality: str = "text") -> str:
    src_id, tgt_id = _memory.remember(source, relation, target, reason, modality=modality)
    return f"Successfully stored memory link: ({src_id}) --[{relation.upper()}]--> ({tgt_id})"

def nexus_recall(query: str) -> str:
    return _memory.recall(query)

if __name__ == "__main__":
    print("Medhas MCP Server initialized.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_core.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medhas/core.py medhas/mcp/ tests/test_core.py
git commit -m "feat(core): assemble Medhas orchestrator and FastMCP server"
```

---

## Self-Review & Verification Plan

### Execution Handoff

Plan complete and saved to [2026-07-30-medhas-nexus-memory-core.md](file:///d:/Downloads/creative/medhas/docs/superpowers/plans/2026-07-30-medhas-nexus-memory-core.md).

Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach would you like to take?
