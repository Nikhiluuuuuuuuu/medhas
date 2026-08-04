"""Perception subsystem (Layer 0 of cognition).

Turns raw multi-modal input into a normalized *perceptual symbol* the rest of the
cognitive stack can reason over. This is the entry point that replaces "the system
only reads text" — it accepts text, structured captions/alt-text, and (as an
extensible adapter interface) image/audio, and emits a canonical ``Percept`` with:

  * extracted entities + relations (reuses the offline graph extractor),
  * a salience score (deterministic: named entities + relation count + novelty vs
    what is already in memory),
  * a modality tag (text / image / audio / structured),
  * a coarse scene/event type inferred from verbs present.

The perception layer is intentionally *model-light and offline-safe*: it does not
require an LLM. Novel modalities (vision/audio) plug in via ``register_adapter``
and only need to return text features (caption/transcript) — the same pipeline then
runs. This keeps the subsystem verifiable and dependency-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agi import entities as _entities
from agi import llm_extract as _extract
from infrastructure.db import DatabasePool


@dataclass
class Percept:
    """A normalized perceptual symbol produced by the perception subsystem."""

    modality: str  # text | image | audio | structured
    raw: str
    entities: List[str] = field(default_factory=list)
    relations: List[tuple] = field(default_factory=list)  # (subj, rel, obj)
    salience: float = 0.0
    scene_type: str = "statement"
    source: str = "perception"

    def to_fact_text(self) -> str:
        """Flatten to a single atomic fact string (for memory ingestion)."""
        if self.relations:
            return "; ".join(f"{s} {r.lower()} {o}" for s, r, o in self.relations)
        return self.raw.strip()


# Optional modality adapters: name -> Callable[[bytes|str], str] returning text features.
_ADAPTERS: Dict[str, Callable[[Any], str]] = {}


def register_adapter(modality: str, fn: Callable[[Any], str]) -> None:
    """Register a perceptual adapter for a new modality (e.g. vision captioner)."""
    _ADAPTERS[modality] = fn


_SCENE_VERBS = {
    "launched": "creation", "founded": "creation", "built": "creation",
    "moved": "relocation", "relocated": "relocation", "traveled": "relocation",
    "prefers": "preference", "likes": "preference", "dislikes": "preference",
    "joined": "affiliation", "works at": "affiliation", "mentors": "social",
    "lives in": "residence", "met": "social", "won": "achievement",
    "failed": "setback", "broke": "failure",
}


def _infer_scene(text: str) -> str:
    low = text.lower()
    for verb, kind in _SCENE_VERBS.items():
        if re.search(rf"(?<![a-z]){re.escape(verb)}(?![a-z])", low):
            return kind
    return "statement"


async def perceive(
    raw: Any,
    modality: str = "text",
    user_id: Optional[str] = None,
    source: str = "perception",
) -> Percept:
    """Convert raw input of a given modality into a Percept.

    Offline-safe: uses the deterministic relation/entity extractor. If an adapter is
    registered for the modality, raw bytes are first turned into text features.
    ``user_id`` (optional) enables a novelty-based salience boost against existing memory.
    """
    if modality in _ADAPTERS:
        text = _ADAPTERS[modality](raw)
    else:
        text = raw if isinstance(raw, str) else str(raw)

    triples, ents = await _extract.extract_graph_open(text)
    names = list({e["name"] for e in ents}) or _entities.extract_entities(text)

    salience = await _salience(text, names, triples, user_id)
    return Percept(
        modality=modality,
        raw=text,
        entities=names,
        relations=triples,
        salience=salience,
        scene_type=_infer_scene(text),
        source=source,
    )


async def _salience(text: str, names: List[str], triples: list, user_id: Optional[str]) -> float:
    """Deterministic salience: entity density + relation count + novelty.

    Novelty: a fact is novel if its entities are not already heavily represented in
    the user's graph (counts of node occurrences). This is the perception-layer
    analogue of 'is this worth attending to' — a core cognitive gating signal.
    """
    base = min(1.0, 0.15 * len(names) + 0.2 * len(triples))
    if not user_id:
        return round(min(1.0, base + 0.1), 3)
    try:
        async with DatabasePool.acquire() as conn:
            existing = 0
            for n in names:
                c = await conn.fetchval(
                    "SELECT count(*) FROM graph_nodes WHERE user_id=$1 AND name ILIKE $2;",
                    user_id, n,
                )
                existing += int(c or 0)
        # Less prior exposure -> higher novelty -> higher salience.
        novelty = 1.0 / (1.0 + existing)
        return round(min(1.0, base + 0.4 * novelty), 3)
    except Exception:
        return round(min(1.0, base + 0.1), 3)


__all__ = ["Percept", "perceive", "register_adapter"]
