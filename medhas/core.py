import time
from typing import Optional, Callable, List
from medhas.storage.kuzu_engine import KuzuStorageEngine
from medhas.storage.vector_index import VectorIndex
from medhas.storage.sqlite_wal import SQLiteWALQueue
from medhas.nlp.canonicalizer import EntityCanonicalizer
from medhas.nlp.multilingual_ai import MultilingualAIExtractor
from medhas.retrieval.spreading_activation import SpreadingActivationEngine
from medhas.retrieval.reranker import RRFFormatter

class MedhasMemoryCore:
    """
    Medhas Dynamic & Multilingual Memory Core Engine.
    Combines LLM Structured Triple Extraction, Multilingual Vector Embeddings,
    Kùzu Knowledge Graph Spreading Activation, and RRF Reranking.
    """
    def __init__(
        self,
        db_path: str = "./medhas_db",
        wal_path: str = "./medhas_wal.db",
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    ):
        self.kuzu = KuzuStorageEngine(db_path)
        self.vector_index = VectorIndex()
        self.wal = SQLiteWALQueue(wal_path)
        self.canonicalizer = EntityCanonicalizer(model_name=model_name)
        self.ai_extractor = MultilingualAIExtractor()
        self.spreading_activation = SpreadingActivationEngine()
        self.formatter = RRFFormatter()
        self._sync_vector_index()

    def _sync_vector_index(self):
        try:
            df = self.kuzu.execute("MATCH (e:Entity) RETURN e.id, e.embedding").get_as_df()
            if not df.empty:
                for _, row in df.iterrows():
                    if row['e.embedding'] is not None:
                        self.vector_index.add_node(row['e.id'], row['e.embedding'], meta={"type": "entity"})
        except Exception:
            pass

    def remember_raw_text(
        self,
        raw_text: str,
        modality: str = "text",
        custom_llm_fn: Optional[Callable[[str], str]] = None
    ) -> list:
        """
        Ingests raw text in ANY language into Medhas memory core.
        1. Indexes raw text chunk embedding into vector index.
        2. Extracts structured SPO triplets via LLM or dynamic OpenIE parser.
        3. Canonicalizes entities and updates Kùzu Knowledge Graph.
        """
        # Index raw text chunk vector for dense retrieval
        chunk_emb = self.canonicalizer.get_embedding(raw_text)
        chunk_id = f"chunk_{hash(raw_text) & 0xffffffff:x}"
        self.vector_index.add_node(chunk_id, chunk_emb, meta={"type": "text_chunk", "text": raw_text})

        triplets = self.ai_extractor.extract_triplets_with_ai(raw_text, custom_llm_fn=custom_llm_fn)
        ingested = []
        for t in triplets:
            src_id, tgt_id = self.remember(
                source=t["source"],
                relation=t["relation"],
                target=t["target"],
                reason=t.get("reason", raw_text),
                modality=modality,
                category_src=t.get("category_src", "MultilingualEntity"),
                category_tgt=t.get("category_tgt", "MultilingualEntity")
            )
            ingested.append((src_id, t["relation"], tgt_id))
        return ingested

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

        try:
            self.kuzu.execute(
                '''
                MATCH (a:Entity {id: $src})-[r:CONNECTS]->(b:Entity {id: $tgt})
                WHERE r.relation <> $rel AND r.valid_to = 0.0
                SET r.valid_to = $now
                ''',
                {"src": src_id, "tgt": tgt_id, "rel": relation.upper(), "now": now}
            )
        except Exception:
            pass

        try:
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
        except Exception:
            pass

        return src_id, tgt_id

    def recall(self, query: str, max_tokens: int = 2000) -> str:
        """
        Recalls relevant facts using hybrid retrieval:
        1. Query dense embedding generation
        2. Kùzu Graph Spreading Activation
        3. Reciprocal Rank Fusion & Reranking
        """
        q_emb = self.canonicalizer.get_embedding(query)
        facts = self.spreading_activation.query(q_emb, self.kuzu, self.vector_index)
        return self.formatter.format_prompt(facts, max_tokens=max_tokens)

