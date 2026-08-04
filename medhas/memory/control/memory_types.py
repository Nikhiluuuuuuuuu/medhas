"""E1 — First-class memory taxonomy + store routing (CoALA + roadmap §0).

Declares every memory type the roadmap requires and the store each type routes to.
Import-only module: no side effects, safe to use anywhere.
"""

from enum import Enum
from typing import Dict


class MemoryType(str, Enum):
    """All memory types Medhas supports (CoALA 4 + roadmap's 6 missing types)."""

    # CoALA core four
    IN_CONTEXT = "in_context"      # working blocks + live turn
    EPISODIC = "episodic"          # raw timestamped events
    SEMANTIC = "semantic"          # distilled facts / gist
    PROCEDURAL = "procedural"      # playbooks / skills

    # Roadmap additions (E27–E31)
    PROSPECTIVE = "prospective"    # future intentions (E27)
    AFFECTIVE = "affective"        # emotionally weighted memory (E28)
    IMPLICIT = "implicit"          # model-inferred, not user-stated (E30)
    META = "meta"                  # knowing what you know (E29)
    SENSORY = "sensory"            # pre-attentive percept buffer (E31)


#: type -> physical store
STORE_ROUTING: Dict[MemoryType, str] = {
    MemoryType.IN_CONTEXT: "working_memory",
    MemoryType.EPISODIC: "episodes",
    MemoryType.SEMANTIC: "atomic_facts",
    MemoryType.PROCEDURAL: "atomic_facts",     # playbooks stored as procedural facts
    MemoryType.PROSPECTIVE: "prospective_memory",
    MemoryType.AFFECTIVE: "atomic_facts",      # affect is a dimension on facts
    MemoryType.IMPLICIT: "atomic_facts",       # tagged via provenance_kind
    MemoryType.META: "meta_memory",
    MemoryType.SENSORY: "percept_buffer",
}

#: types that live in the atomic_facts table and are retrievable by search_facts
FACT_BACKED_TYPES = {
    MemoryType.SEMANTIC,
    MemoryType.EPISODIC,
    MemoryType.PROCEDURAL,
    MemoryType.AFFECTIVE,
    MemoryType.IMPLICIT,
}


def route(memory_type: str) -> str:
    """Return the physical store name for a memory type string."""
    try:
        return STORE_ROUTING[MemoryType(memory_type)]
    except (ValueError, KeyError):
        return "atomic_facts"


def is_valid(memory_type: str) -> bool:
    try:
        MemoryType(memory_type)
        return True
    except ValueError:
        return False
