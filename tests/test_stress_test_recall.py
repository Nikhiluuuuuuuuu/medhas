import pytest
import shutil
from medhas.core import MedhasMemoryCore

@pytest.fixture
def temp_core(tmp_path):
    db_dir = str(tmp_path / "stress_kuzu")
    wal_path = str(tmp_path / "stress_wal.db")
    core = MedhasMemoryCore(db_path=db_dir, wal_path=wal_path)
    yield core
    shutil.rmtree(db_dir, ignore_errors=True)

def test_kyoto_stress_test(temp_core):
    text = (
        "Yesterday at 3:15 PM under the weeping cherry tree in Kyoto's Maruyama Park, "
        "Dr. Elena Rostova, a 42-year-old astrophysicist from Zurich who drinks Earl Grey tea "
        "and owns a tri-color Pembroke Welsh Corgi named Orion, traded a silver 1958 Leica M3 camera—"
        "which she inherited from her maternal grandfather, Arthur Pendelton—to her younger brother Marcus, "
        "a Toronto-based architect with a shellfish allergy, in exchange for a hand-bound leather journal "
        "containing secret 1924 baking recipes written by their great-aunt Beatrice, who married an Italian "
        "clockmaker named Matteo in Venice."
    )

    ingested = temp_core.remember_raw_text(text)
    assert len(ingested) > 0, "Should ingest multiple memory graph links"

    # Verify no invalid "who married" subject nodes were extracted
    for src, rel, tgt in ingested:
        assert src != "who married", "Should not extract 'who married' as subject node"

    # Query 1: Marcus's sister profile
    q1 = temp_core.recall("What is the name, age, profession, home city, preferred tea, and pet (breed and name) of Marcus's sister?")
    assert len(q1) > 0

    # Query 2: Camera provenance & ownership
    q2 = temp_core.recall("Who originally owned the silver 1958 Leica M3 camera, how did Elena get it, and who owns it after the trade?")
    assert len(q2) > 0

    # Query 3: Kinship & Relations
    q3 = temp_core.recall("What is the exact family relationship between Matteo and Marcus, and where was Matteo married?")
    assert len(q3) > 0

    # Query 4: Spatio-temporal context
    q4 = temp_core.recall("At what time, under what specific landmark tree, and in which park and city did the transaction take place?")
    assert len(q4) > 0

    # Query 5: Multi-hop connection
    q5 = temp_core.recall("What medical allergy does the new owner of the Leica M3 camera have, and where does he live?")
    assert len(q5) > 0

    # Query 6: Object properties
    q6 = temp_core.recall("What did Marcus give away in the trade, who authored its contents, and in what year were those contents written?")
    assert len(q6) > 0
