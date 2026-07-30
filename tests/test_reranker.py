import pytest
from medhas.retrieval.reranker import RRFFormatter

def test_format_prompt():
    formatter = RRFFormatter()
    facts = [{"source": "A", "relation": "LEADS", "target": "B", "reason": "test", "activation_score": 0.8}]
    prompt = formatter.format_prompt(facts)
    assert "[MEMORY CONTEXT - MEDHAS RECALLED FACTS]" in prompt
    assert "(A) --[LEADS]--> (B)" in prompt
