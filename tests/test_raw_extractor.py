import pytest
import shutil
from medhas.core import MedhasMemoryCore

@pytest.fixture
def temp_core(tmp_path):
    db_dir = str(tmp_path / "multilingual_kuzu")
    wal_path = str(tmp_path / "multilingual_wal.db")
    core = MedhasMemoryCore(db_path=db_dir, wal_path=wal_path)
    yield core
    shutil.rmtree(db_dir, ignore_errors=True)

def test_multilingual_ai_extraction(temp_core):
    # Hindi Text (हिंदी): "एलिस स्मिथ प्रोजेक्ट टाइटन का नेतृत्व करती हैं।"
    hindi_text = "एलिस स्मिथ प्रोजेक्ट टाइटन का नेतृत्व करती हैं।"
    temp_core.remember_raw_text(hindi_text)

    # Spanish Text (Español): "El proyecto Titán está bloqueado por la migración."
    spanish_text = "El proyecto Titán está bloqueado por la migración."
    temp_core.remember_raw_text(spanish_text)

    # French Text (Français): "La migration nécessite une documentation."
    french_text = "La migration nécessite une documentation."
    temp_core.remember_raw_text(french_text)

    # Verify multilingual memory recall
    recalled_hindi = temp_core.recall("एलिस स्मिथ")
    assert len(recalled_hindi) > 0

    recalled_spanish = temp_core.recall("Titán")
    assert len(recalled_spanish) > 0
