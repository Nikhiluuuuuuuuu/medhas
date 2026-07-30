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
