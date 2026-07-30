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
    df_node = engine.execute("MATCH (e:Entity) RETURN e.id").get_as_df()
    assert df_node.empty
