"""G2 — Dependency-free entity + relation extraction for graph memory.

No external NER (spaCy/nltk not installed; project must stay self-contained). We use a
robust heuristic tuned for the memory domain:

  * Entities  : capitalized tokens/token-groups (person / org / product / place / date),
                plus a small dictionary of known relationship verbs to split clauses.
  * Relations : simple subject–verb–object triples derived from clause splits on
                copular ("is", "are", "was", "were") and possessive / action verbs
                ("founded", "builds", "lives in", "co-founded", "moved", ...).

The extractor is intentionally light: its job is to populate the existing graph
(upsert_node / update_edge / query_subgraph) so recall can do 2-hop traversal.
"""

from typing import Dict, List, Tuple, Optional

# Relationship verbs we turn into typed edges (subject --[rel]--> object).
# Extended well beyond the original small list so open extraction works WITHOUT an
# LLM (offline mode): natural English verbs like "launched", "mentors", "makes",
# "joined", "relocated" are handled by the deterministic heuristic, not a cloud call.
RELATION_VERBS = {
    "founded": "FOUNDED",
    "co-founded": "CO_FOUNDED",
    "cofounded": "CO_FOUNDED",
    "builds": "BUILDS",
    "build": "BUILDS",
    "made": "MADE",
    "makes": "MAKES",
    "make": "MAKES",
    "created": "CREATED",
    "create": "CREATES",
    "launched": "LAUNCHED",
    "launch": "LAUNCHED",
    "released": "RELEASED",
    "mentors": "MENTORS",
    "mentor": "MENTORS",
    "advises": "ADVISES",
    "advise": "ADVISES",
    "teaches": "TEACHES",
    "joined": "JOINED",
    "join": "JOINED",
    "works at": "WORKS_AT",
    "works for": "WORKS_FOR",
    "employed by": "WORKS_FOR",
    "relocated to": "RELOCATED_TO",
    "relocated": "RELOCATED_TO",
    "moved to": "MOVED_TO",
    "moved": "MOVED_TO",
    "lives in": "LIVES_IN",
    "live in": "LIVES_IN",
    "lived in": "LIVED_IN",
    "is": "IS_A",
    "was": "IS_A",
    "are": "IS_A",
    "were": "IS_A",
    "prefers": "PREFERS",
    "prefer": "PREFERS",
    "uses": "USES",
    "use": "USES",
    "owns": "OWNS",
    "own": "OWNS",
    "leads": "LEADS",
    "lead": "LEADS",
    "runs": "RUNS",
    "headquarters": "HEADQUARTERS",
    "based in": "BASED_IN",
    "located in": "LOCATED_IN",
    "makes": "MAKES",
}

COPULAR = {"is", "are", "was", "were", "been"}

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


def extract_relations(text: str) -> List[Tuple[str, str, str]]:
    """Return (subject, relationship, object) triples derived from clause splits.

    Only emits a triple when both subject and object are non-empty capitalized spans
    (or date/number literals), so the graph stays clean. Verbs are matched on WORD
    BOUNDARIES so substrings like 'are' inside 'KareOS' don't create bogus triples.
    """
    import re
    triples: List[Tuple[str, str, str]] = []
    lowered = text.lower()

    # Longest verb match first (so 'co-founded' beats 'founded', 'lives in' beats 'lives')
    verbs = sorted(RELATION_VERBS.keys(), key=len, reverse=True)
    for verb in verbs:
        # word-boundary match: avoid matching 'are' inside 'KareOS', 'is' inside 'kraionyx'
        pattern = r"(?<![a-z0-9])" + re.escape(verb) + r"(?![a-z0-9])"
        for m in re.finditer(pattern, lowered):
            idx = m.start()
            subj_part = text[:idx].strip()
            obj_part = text[idx + len(verb):].strip()
            subj = _last_entity(subj_part)
            obj = _first_entity(obj_part)
            if subj and obj and subj.lower() != obj.lower():
                triples.append((subj, RELATION_VERBS[verb], obj))
    return triples


def _last_entity(text: str) -> Optional[str]:
    ents = extract_entities(text)
    if ents:
        return ents[-1]
    # Offline fallback: grab the last alphabetic word (skip year/number tokens like
    # '2023', which are fact properties, not entities). Lowercase names like 'priya'
    # are valid subjects even though the heuristic NER only matches capitals. This is
    # what lets open extraction work WITHOUT an LLM for natural-language lowercase names.
    words = [w.strip(".,!?:;\"'()[]") for w in text.replace(",", " ").split()]
    words = [w for w in words if w and w.lower() not in _STOPWORDS and not w.isdigit()
             and not (len(w) == 4 and w.isdigit())]
    return words[-1] if words else None


def _first_entity(text: str) -> Optional[str]:
    # for object, take the first capitalized span; fall back to first alphabetic token
    ents = extract_entities(text)
    if ents:
        return ents[0]
    toks = [t.strip(".,!?:;\"'()[]") for t in text.replace(",", " ").split()]
    toks = [t for t in toks if t and t.lower() not in _STOPWORDS and not t.isdigit()]
    return toks[0] if toks else None


def query_entities(query: str) -> List[str]:
    """Entities mentioned in a query — used to seed multi-hop graph traversal."""
    return extract_entities(query)


__all__ = [
    "extract_entities", "extract_relations", "query_entities",
    "RELATION_VERBS",
]
