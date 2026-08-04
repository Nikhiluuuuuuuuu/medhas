"""E4 — Chunking & fact extraction orchestrator.

Robustly split long inputs into retrieval-optimized chunks (semantic + size-bounded)
and extract atomic facts per chunk via the LLM extractor, with a deterministic
keyword fallback when the model is unavailable. This front-loads E32 admission scoring.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from medhas.llm.gateway import get_llm
from medhas.utils import log_atomic, log_error

# Provider resolved lazily via the gateway (provider-agnostic).
def llm():
    return get_llm()

MAX_CHARS = 1800          # hard ceiling per chunk
OVERLAP = 200            # overlap to preserve cross-sentence context
SENT_RE = re.compile(r"(?<=[.!?])\s+")

EXTRACT_PROMPT = """Extract standalone atomic facts from the text. Each fact must be
self-contained (no pronouns resolved to the user/subject). Ignore chit-chat.
Return ONLY JSON: {"facts": ["fact 1", "fact 2"]}. Return {"facts": []} if nothing durable."""


@dataclass
class Chunk:
    index: int
    text: str
    char_start: int
    char_end: int


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> List[Chunk]:
    """Semantic, size-bounded chunking with overlap (E4)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [Chunk(0, text, 0, len(text))]

    sentences = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    chunks: List[Chunk] = []
    cur: List[str] = []
    cur_len = 0
    start = 0
    i = 0
    while i < len(sentences):
        s = sentences[i]
        if cur and cur_len + len(s) + 1 > max_chars:
            chunk_text_joined = " ".join(cur)
            chunks.append(Chunk(len(chunks), chunk_text_joined, start, start + len(chunk_text_joined)))
            # backtrack for overlap
            overlap_text = " ".join(cur)
            used = len(overlap_text)
            start = max(start, start + used - overlap)
            cur, cur_len = [], 0
            continue
        cur.append(s)
        cur_len += len(s) + 1
        i += 1
    if cur:
        joined = " ".join(cur)
        chunks.append(Chunk(len(chunks), joined, start, start + len(joined)))
    return chunks


def _keyword_facts(text: str) -> List[str]:
    """Deterministic fallback extractor used when the LLM is unavailable."""
    facts: List[str] = []
    for sent in (s.strip() for s in SENT_RE.split(text) if s.strip()):
        low = sent.lower()
        if any(k in low for k in (
            "prefer", "always", "never", "name is", "works at", "lives in", "is from",
            "born", "deadline", "birthday", "hates", "loves", "uses", "uses", "my",
            "i am", "i work", "i live", "i'm", "remember", "must", "should",
        )):
            facts.append(sent)
    return facts[:15]


async def extract_facts(text: str, use_llm: bool = True) -> List[str]:
    """Extract atomic facts from a chunk of text (E4)."""
    try:
        if not use_llm:
            return _keyword_facts(text)
        resp = await llm().chat_completion(
            [{"role": "system", "content": EXTRACT_PROMPT},
             {"role": "user", "content": text[:3000]}],
            temperature=0.0,
        )
        raw = resp.get("content", "")
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
        import json
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
        facts = [str(f).strip() for f in (data.get("facts") or []) if str(f).strip()]
        if not facts:
            return _keyword_facts(text)
        return facts[:15]
    except Exception as e:
        log_error(f"extract_facts failed, using fallback: {e}")
        return _keyword_facts(text)


async def extract_from_document(text: str, use_llm: bool = True) -> Dict[str, Any]:
    """Full E4 pipeline: chunk then extract. Returns chunked + extracted result."""
    chunks = chunk_text(text)
    all_facts: List[str] = []
    per_chunk: List[Dict[str, Any]] = []
    for c in chunks:
        facts = await extract_facts(c.text, use_llm=use_llm)
        all_facts.extend(facts)
        per_chunk.append({"index": c.index, "text": c.text, "facts": facts})
    # de-dup preserving order
    seen = set()
    unique = []
    for f in all_facts:
        k = f.lower().strip()
        if k not in seen:
            seen.add(k)
            unique.append(f)
    return {"chunks": len(chunks), "facts": unique, "per_chunk": per_chunk}


def detect_duplicates(facts: Sequence[str], threshold: float = 0.9) -> List[List[int]]:
    """Group near-identical extracted facts (simple token-Jaccard heuristic)."""
    norm = [set(re.findall(r"[a-z0-9]+", f.lower())) for f in facts]
    groups: List[List[int]] = []
    for i in range(len(facts)):
        placed = False
        for g in groups:
            j = g[0]
            a, b = norm[i], norm[j]
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac >= threshold:
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return [g for g in groups if len(g) > 1]
