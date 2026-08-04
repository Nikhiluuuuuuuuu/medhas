"""Open (non-keyword) extraction + resolution for graph memory.

LLM-driven open extraction: the model emits *any* (subject, relationship, object)
triples it can find, plus entity types, regardless of wording. There is NO closed
vocabulary and NO hard-coded relation list — new relations, lowercase names, and novel
phrasings are all handled by the model. Each discovered relation is recorded into the
evolving `relation_types` vocabulary (agi.cognition.schema_evolution) so the edge-type
set grows with the data rather than being capped by a frozen enum.

This module requires a working LLM provider (Groq). If the LLM call fails for any
reason, the functions return empty results (no relation triples invented) so callers
can degrade gracefully (e.g. store the raw turn as an atomic fact) instead of silently
fabricating hard-coded graph edges.
"""

from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timezone

from infrastructure.llm import GroqLLMProvider
from utils import log_error, measure_latency

# Shared provider instance (cheap; the underlying client is lazy / cached)
_llm = GroqLLMProvider()

_EXTRACT_SYS = (
    "You are an expert knowledge-graph extraction specialist for an AI agent memory system. "
    "Extract ALL subject-verb-object relations as typed triples, plus entity types, from ONE sentence. "
    "Output ONLY valid JSON: "
    '{"triples":[["subject","RELATIONSHIP","object"], ...], '
    '"entities":[{"name":"...","type":"PERSON|ORG|PRODUCT|PLACE|TOOL|CONCEPT|ATTR|EVENT"}]}. '
    "RELATIONSHIP names: UPPER_SNAKE_CASE describing the action/link in plain English "
    "(e.g. CO_FOUNDED, BUILDS, LAUNCHED, ADVISES, MENTORS, LIVES_IN, PREFERS, ACQUIRED, "
    "WORKS_AT, IS_CEO_OF, WON, PUBLISHED). Keep entity names EXACTLY as written, including "
    "lowercase or single-word names. "
    "ENTITY RULES (critical): "
    "1. Entity names must be NOUN PHRASES (<=5 words) referring to a discrete thing in the "
    "user's world. "
    "2. Do NOT extract as entities: pure dates/years or numeric quantities (e.g. '2023', "
    "'in 2024', '3 times', '$150'); pronouns or unresolved references ('he', 'the company' "
    "without a name, 'this issue'); geographic coordinates; specific clock times. These are "
    "PROPERTIES of a fact, not entities — keep them inside the fact/object text. "
    "3. Extract projects, products, pets, and creations as SEPARATE entities, not just the "
    "person. "
    "FACT RULES: "
    "4. Each triple's subject and object must be a named entity from your entity list; never "
    "collapse two entities into a self-reference unless no second entity fits. "
    "5. Facts must be SELF-CONTAINED — understandable without the original sentence. Preserve "
    "specific details (dates, places, numbers) inside the object or fact text. "
    "6. Prefer DIRECT subject-to-target edges; do not route a relationship through descriptive "
    "scenery nouns. "
    "Do not invent facts not in the sentence. If none, return empty arrays."
)

_RESOLVE_SYS = (
    "You resolve vague references in a question to concrete entity names that exist in the "
    "user's memory graph. Given a question and a JSON list of known entity names, return ONLY "
    "JSON: {\"entities\":[<resolved name>, ...]}. "
    "Rules: "
    "1. Replace pronouns/definite descriptions with the best matching known entity "
    "(e.g. 'he' -> the most salient person; 'the company that builds X' -> the builder of X; "
    "'it'/'they' -> the most relevant recent entity). "
    "2. Only output names from the provided list. "
    "3. Resolve 2-hop descriptions via the graph: if the question names an entity that is "
    "linked to another (e.g. 'the product that makes solar panels' where the product is linked "
    "to a maker), include BOTH the named entity and its linked counterpart. "
    "4. If nothing resolves, output the original capitalized names found in the question."
)


def _safe_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # tolerate ```json fences or trailing prose
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                return None
    return None


async def extract_graph_open(
    fact_text: str, user_id: Optional[str] = None
) -> Tuple[List[Tuple[str, str, str]], List[Dict[str, str]]]:
    """LLM-based OPEN extraction. Returns (triples, entity_type_hints).

    Relations are NOT taken from a closed hard-coded list: the LLM may emit ANY
    relationship string, and each discovered relation is recorded into the evolving
    `relation_types` vocabulary (schema evolution) when ``user_id`` is provided.

    Requires a working LLM provider. On any LLM failure the function returns empty
    results (no relation triples invented) so callers can degrade gracefully (e.g. store
    the raw turn as an atomic fact) instead of fabricating hard-coded graph edges.
    """
    async with measure_latency("agi.llm_extract.extract_graph_open"):
        # Vocabulary hint: let the LLM reuse existing relations where semantically apt,
        # which reduces synonym drift (e.g. CO_FOUNDED vs COFOUNDED) without capping it.
        vocab_hint = ""
        if user_id:
            try:
                from agi.cognition.schema_evolution import known_relations
                known = await known_relations(user_id)
                if known:
                    vocab_hint = (
                        " Prefer reusing an EXISTING relation from this user's vocabulary "
                        f"when semantically identical: {sorted(known)[:40]}. "
                        "Otherwise coin a clear new UPPER_SNAKE_CASE relation."
                    )
            except Exception:
                pass
        try:
            sys_prompt = _EXTRACT_SYS + vocab_hint
            resp = await _llm.chat_completion([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": fact_text},
            ], temperature=0.0)
            data = _safe_json(resp.get("content", "")) if isinstance(resp, dict) else None
            if not data:
                raise ValueError("empty/non-json extraction")
            triples = []
            for t in data.get("triples", []):
                if isinstance(t, (list, tuple)) and len(t) == 3:
                    s, r, o = (str(x).strip() for x in t)
                    if s and r and o and s.lower() != o.lower():
                        triples.append((s, r.upper(), o))
            ents = []
            for e in data.get("entities", []):
                if isinstance(e, dict) and e.get("name"):
                    ents.append({"name": str(e["name"]).strip(),
                                 "type": str(e.get("type", "ENTITY")).upper()})
            if triples or ents:
                # OPEN vocabulary: record every discovered relation into the evolving set.
                if user_id:
                    from agi.cognition.schema_evolution import record_relation
                    for _s, r, _o in triples:
                        await record_relation(user_id, r, source="extracted")
                return triples, ents
            raise ValueError("no triples/entities produced")
        except Exception as e:
            # No hard-coded fallback: return empty results, never invent graph edges.
            # Callers (async_extractor / graph_build) degrade by storing the raw turn.
            log_error(f"extract_graph_open LLM failed, returning empty (no fallback invent): {e}")
            return [], []


async def extract_date_open(fact_text: str) -> Optional[datetime]:
    """LLM-based date extraction -> ISO date (YYYY-MM-DD) or None.

    Handles relative/implicit phrasing better than pure regex. Falls back to the
    regex extractor (utils.dates) on failure. Relative prepositions ("before/after
    <year>") are resolved to an adjacent year so temporal ordering works even when
    both events share the anchor year (e.g. "lived in chennai before moving to
    bengaluru in 2024" -> chennai gets valid_from 2023).
    """
    async with measure_latency("agi.llm_extract.extract_date_open"):
        try:
            resp = await _llm.chat_completion([
                {"role": "system", "content": (
                    "Extract the single most specific absolute date mentioned in the "
                    "sentence as an ISO date (YYYY-MM-DD). If a year only, use Jan 1. "
                    "If no date is stated, reply with the single word NONE. If the sentence "
                    "uses a relative anchor like 'before <year>' or 'after <year>' with no "
                    "own absolute date for that clause, return the adjacent year Jan 1 "
                    "(before -> year-1, after -> year+1). Reply with ONLY the date or NONE, "
                    "no explanation.")},
                {"role": "user", "content": fact_text},
            ], temperature=0.0)
            content = (resp.get("content", "") if isinstance(resp, dict) else "").strip()
            if not content or content.upper() == "NONE":
                # regex fallback may still find an anchor year
                return _regex_date_or_relative(fact_text)
            import re
            m = re.search(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", content)
            if not m:
                raise ValueError("no date in llm reply")
            y, mo, d = m.groups()
            mo = mo or "01"; d = d or "01"
            from datetime import datetime, timezone
            return datetime(int(y), int(mo), int(d), tzinfo=timezone.utc)
        except Exception as e:
            log_error(f"llm_date skipped, using regex: {e}")
            return _regex_date_or_relative(fact_text)


def _regex_date_or_relative(fact_text: str) -> Optional[datetime]:
    """Regex anchor + relative-preposition resolution (offline fallback)."""
    from utils.dates import extract_fact_date
    dt = extract_fact_date(fact_text)
    if dt:
        return dt
    import re
    # "before/after <event> in <YEAR>" (e.g. "lived in chennai before moving to
    # bengaluru in 2024") -> the earlier clause gets YEAR-1 so temporal ordering
    # works even when the earlier clause itself states no year. This is the H1 fix:
    # relative residence/employment before a dated event must be orderable.
    m = re.search(r"\b(before|prior to|after)\b[^.]*?\b(19|20)\d{2}\b", fact_text.lower())
    if m:
        direction = m.group(1)
        year = int(re.search(r"(19|20)\d{2}", m.group(0)).group(0))
        y = year - 1 if direction in ("before", "prior to") else year + 1
        from datetime import datetime, timezone
        return datetime(y, 1, 1, tzinfo=timezone.utc)
    # "before <year>" / "after <year>" with no own date -> adjacent year
    m = re.search(r"\b(before|after|prior to)\s+(\d{4})\b", fact_text.lower())
    if m:
        year = int(m.group(2))
        y = year - 1 if m.group(1) in ("before", "prior to") else year + 1
        from datetime import datetime, timezone
        return datetime(y, 1, 1, tzinfo=timezone.utc)
    # bare "<year>" anchor
    m2 = re.search(r"\b(19|20)\d{2}\b", fact_text)
    if m2:
        from datetime import datetime, timezone
        return datetime(int(m2.group(0)), 1, 1, tzinfo=timezone.utc)
    return None


async def resolve_entities_open(query: str, known_entities: List[str]) -> List[str]:
    """LLM-based anaphora/coreference resolution against known graph entities.

    On LLM failure, degrades to capitalized-token matching against the query plus
    containment in the known-entity set (no hard-coded relation words invented).
    """
    async with measure_latency("agi.llm_extract.resolve_entities_open"):
        try:
            if not known_entities:
                raise ValueError("no known entities")
            resp = await _llm.chat_completion([
                {"role": "system", "content": _RESOLVE_SYS},
                {"role": "user", "content": f"KNOWN ENTITIES: {json.dumps(known_entities)}\nQUESTION: {query}"},
            ], temperature=0.0)
            data = _safe_json(resp.get("content", "")) if isinstance(resp, dict) else None
            if data and data.get("entities"):
                return [str(x).strip() for x in data["entities"] if x]
            raise ValueError("no resolution produced")
        except Exception as e:
            log_error(f"llm_resolve failed, using capitalized-token fallback: {e}")
            from agi.entities import query_entities
            # Graceful degradation: (1) direct capitalized entities in the query, then
            # (2) match against the provided known_entities by word/lowercase containment
            # so lowercase known names (e.g. 'priya') still resolve without the LLM.
            resolved = list(query_entities(query))
            ql = query.lower()
            for name in known_entities:
                if not name:
                    continue
                nl = name.lower()
                if nl in ql and name not in resolved:
                    resolved.append(name)
            return resolved


__all__ = ["extract_graph_open", "extract_date_open", "resolve_entities_open"]
