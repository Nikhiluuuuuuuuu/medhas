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
