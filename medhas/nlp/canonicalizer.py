import os
import time
import rapidfuzz
import numpy as np
from typing import Optional, List

class EntityCanonicalizer:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.encoder = None
        self._init_encoder()

    def _init_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.encoder = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer
                # Enable online fetch if local_files_only fails
                if "HF_HUB_OFFLINE" in os.environ:
                    del os.environ["HF_HUB_OFFLINE"]
                self.encoder = SentenceTransformer(self.model_name)
            except Exception:
                self.encoder = None

    def clean_name(self, name: str) -> str:
        return name.strip()

    def get_embedding(self, text: str) -> list:
        if self.encoder is not None:
            try:
                emb = self.encoder.encode(text).tolist()
                return emb
            except Exception:
                pass
        
        vec = np.zeros(384, dtype=np.float32)
        clean = self.clean_name(text).lower()
        for i, char in enumerate(clean):
            idx = (ord(char) * (i + 1)) % 384
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def canonicalize(self, raw_name: str, category: str, kuzu_engine, vector_index) -> str:
        name = self.clean_name(raw_name)
        if not name:
            return "Entity"
        now = time.time()
        
        # 1. Direct match check in KuzuDB persistent DB
        try:
            df_exist = kuzu_engine.execute(
                "MATCH (e:Entity {id: $id}) RETURN e.id, e.embedding", 
                {"id": name}
            ).get_as_df()
            
            if not df_exist.empty:
                kuzu_engine.execute(
                    "MATCH (e:Entity {id: $id}) SET e.last_accessed = $now, e.access_count = e.access_count + 1",
                    {"id": name, "now": now}
                )
                emb = df_exist['e.embedding'].iloc[0]
                if emb is not None:
                    vector_index.add_node(name, emb, meta={"type": "entity"})
                return name
        except Exception:
            pass

        # 2. Vector & Fuzzy similarity search
        new_emb = self.get_embedding(name)
        matches = vector_index.search(new_emb, top_k=5, item_type="entity")
        for cand_id, sim in matches:
            if rapidfuzz.fuzz.ratio(name.lower(), cand_id.lower()) > 88 or sim > 0.88:
                try:
                    kuzu_engine.execute(
                        "MATCH (e:Entity {id: $id}) SET e.last_accessed = $now, e.access_count = e.access_count + 1",
                        {"id": cand_id, "now": now}
                    )
                except Exception:
                    pass
                return cand_id

        # 3. Create new entity node safely
        try:
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
            vector_index.add_node(name, new_emb, meta={"type": "entity"})
        except Exception:
            # Entity already exists
            try:
                kuzu_engine.execute(
                    "MATCH (e:Entity {id: $id}) SET e.last_accessed = $now, e.access_count = e.access_count + 1",
                    {"id": name, "now": now}
                )
            except Exception:
                pass
            vector_index.add_node(name, new_emb, meta={"type": "entity"})

        return name

