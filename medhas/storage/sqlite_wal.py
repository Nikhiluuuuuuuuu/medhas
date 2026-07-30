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

    def get_pending_facts(self) -> List[Dict[str, Any]]:
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
