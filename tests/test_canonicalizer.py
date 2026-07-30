import pytest
from medhas.nlp.canonicalizer import EntityCanonicalizer

def test_canonicalization_merge():
    canon = EntityCanonicalizer()
    res1 = canon.clean_name(" Alice Smith ")
    res2 = canon.clean_name("Alice Smith")
    assert res1 == res2
