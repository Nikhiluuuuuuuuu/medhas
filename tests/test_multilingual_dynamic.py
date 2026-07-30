import os
import pytest
import tempfile
import shutil
from medhas.core import MedhasMemoryCore
from medhas.nlp.raw_extractor import UniversalDynamicExtractor
from medhas.nlp.multilingual_ai import MultilingualAIExtractor

@pytest.fixture
def temp_db_dir():
    dir_path = tempfile.mkdtemp(prefix="medhas_test_")
    wal_path = os.path.join(dir_path, "test_wal.db")
    db_path = os.path.join(dir_path, "test_kuzu")
    yield db_path, wal_path
    shutil.rmtree(dir_path, ignore_errors=True)

def test_dynamic_extractor_no_hardcoded_clause():
    extractor = UniversalDynamicExtractor()
    text = "Elena Rostova lives in Zurich and she owns Orion the Corgi."
    triplets = extractor.extract_triplets(text)
    
    assert len(triplets) > 0
    for t in triplets:
        assert not t["source"].lower().startswith("who married")
        assert len(t["source"]) > 0
        assert len(t["target"]) > 0

def test_multilingual_triplet_extraction():
    ai_extractor = MultilingualAIExtractor()
    
    # English
    en_text = "Nikhil Sai is the founder of Medhas and he lives in Hyderabad."
    triplets_en = ai_extractor.extract_triplets_with_ai(en_text)
    assert len(triplets_en) >= 1

    # Spanish
    es_text = "Elena Rostova vive en Zurich y trabaja como ingeniera."
    triplets_es = ai_extractor.extract_triplets_with_ai(es_text)
    assert len(triplets_es) >= 1

def test_medhas_core_remember_and_recall(temp_db_dir):
    db_path, wal_path = temp_db_dir
    memory = MedhasMemoryCore(db_path=db_path, wal_path=wal_path)

    # Ingest narrative memory
    memory.remember_raw_text("Dr. Elena Rostova is a quantum physicist living in Zurich.")
    memory.remember_raw_text("Dr. Elena Rostova owns a Pembroke Welsh Corgi named Orion.")

    # Recall in English
    context_en = memory.recall("Where does Elena Rostova live?")
    assert "Zurich" in context_en or "Elena" in context_en

    # Recall query
    context_corgi = memory.recall("What dog does Elena own?")
    assert "Orion" in context_corgi or "Corgi" in context_corgi or "Elena" in context_corgi

def test_llm_structured_extraction():
    extractor = UniversalDynamicExtractor()
    
    # Mock LLM function simulating JSON response
    def mock_llm_fn(prompt: str) -> str:
        return '''
        [
            {"source": "Elena Rostova", "relation": "MARRIED_TO", "target": "Matteo", "category_src": "Person", "category_tgt": "Person"},
            {"source": "Matteo", "relation": "PROFESSION", "target": "Clockmaker", "category_src": "Person", "category_tgt": "Profession"}
        ]
        '''
    
    text = "Elena Rostova married Matteo, who is a clockmaker."
    triplets = extractor.extract_triplets(text, custom_llm_fn=mock_llm_fn)
    
    assert len(triplets) == 2
    assert triplets[0]["source"] == "Elena Rostova"
    assert triplets[0]["relation"] == "MARRIED_TO"
    assert triplets[0]["target"] == "Matteo"
    assert triplets[1]["source"] == "Matteo"
    assert triplets[1]["relation"] == "PROFESSION"
    assert triplets[1]["target"] == "Clockmaker"
