"""Shared lightweight entity helpers for graph memory.

This module provides ONLY dependency-free capitalized entity recognition
(`extract_entities` / `query_entities`). It deliberately contains NO hard-coded
relation vocabulary: relation/edge extraction is performed by the LLM in
`agi.llm_extract.extract_graph_open`, which emits any relation string and records
it into the evolving `relation_types` vocabulary (agi.cognition.schema_evolution).

The historical `RELATION_VERBS` dictionary + `extract_relations` rule-based triple
extractor were removed: a closed verb list is the anti-pattern the field warns
against (Graphiti/Zep, Mem0, Letta all use open LLM-driven relation extraction).
"""

from typing import List, Optional

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "with", "for", "of", "to", "in", "on",
    "at", "by", "from", "that", "this", "these", "those", "his", "her", "their",
    "our", "your", "as", "into", "before", "after", "when", "where", "what",
    "who", "which", "how", "why", "he", "she", "it", "they", "we", "you", "i",
}


def _capital_entities(text: str) -> List[str]:
    """Pull multi-word capitalized spans (e.g. 'Kraionyx AI', 'Nikhil', 'KOS-7')."""
    tokens = text.split()
    ents: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i].strip(".,!?:;\"'()[]")
        # accept tokens that start uppercase and are alphanumeric (allowing internal -/_/digits
        # so 'KOS-7', 'GPT-4', 'BAAI/bge' stay whole)
        _tok_ok = (
            tok[:1].isupper()
            and tok.lower() not in _STOPWORDS
            and any(ch.isalnum() for ch in tok)
            and all(ch.isalnum() or ch in "-_/." for ch in tok)
        )
        if _tok_ok:
            span = [tok]
            j = i + 1
            while j < len(tokens):
                nxt = tokens[j].strip(".,!?:;\"'()[]")
                _nxt_ok = (
                    (nxt[:1].isupper() or nxt.lower() in {"ai", "os", "inc", "labs", "gmbh", "llc", "and"})
                    and any(ch.isalnum() for ch in nxt)
                    and all(ch.isalnum() or ch in "-_/." for ch in nxt)
                )
                if _nxt_ok:
                    span.append(nxt)
                    j += 1
                else:
                    break
            ent = " ".join(span).strip(".,!?:;\"'()[]")
            if ent:
                ents.append(ent)
            i = j if j > i + 1 else i + 1
        else:
            i += 1
    return ents


def extract_entities(text: str) -> List[str]:
    """Return unique capitalized entity names found in text (order-preserving)."""
    seen = set()
    out = []
    for e in _capital_entities(text):
        key = e.lower()
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def query_entities(query: str) -> List[str]:
    """Entities mentioned in a query — used to seed/extend graph traversal candidates."""
    return extract_entities(query)


__all__ = [
    "extract_entities",
    "query_entities",
]
