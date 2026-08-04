"""Generalization subsystem (schema induction + analogical transfer).

Cognition is not just storing facts — it is *abstracting* over them and *reusing* the
abstraction on novel cases. This module implements two real, offline, verifiable abilities:

  1. ``induce_schema(instances)`` — from observed (subject, rel, object) triples, induce a
     typed relation schema: which entity *types* typically play subject vs object, and a
     confidence from support count. This is few-shot schema induction (the basis of
     "I've seen this pattern before, so I know how to handle a new instance").

  2. ``apply_schema(schema, novel_pair)`` — given an induced schema and a NEW (subject, object)
     pair, predict the missing relation (a form of analogical completion).

  3. ``analogy(a, b, graph)`` — map a known relational structure centered on entity ``a``
     onto a new entity ``b``: "A relates to X like B relates to Y". This is structural
     analogy (the cognitive core of transfer learning), done by copying the relation
     skeleton from one node's neighborhood to another.

All three are deterministic and tested against the live graph DB.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from infrastructure.db import DatabasePool

Triple = Tuple[str, str, str]


@dataclass
class RelationSchema:
    relation: str
    subject_types: Dict[str, int] = field(default_factory=dict)
    object_types: Dict[str, int] = field(default_factory=dict)
    support: int = 0

    def confidence(self) -> float:
        return min(1.0, self.support / 3.0)  # needs >=3 examples to be "reliable"

    def predict_object_type(self) -> str:
        if not self.object_types:
            return "ENTITY"
        return max(self.object_types, key=lambda k: self.object_types[k])


def induce_schema(instances: List[Triple]) -> Dict[str, RelationSchema]:
    """Induce a typed relation schema from observed triples (few-shot)."""
    out: Dict[str, RelationSchema] = {}
    for s, r, o in instances:
        sch = out.setdefault(r, RelationSchema(relation=r))
        sch.subject_types[_guess_type(s)] = sch.subject_types.get(_guess_type(s), 0) + 1
        sch.object_types[_guess_type(o)] = sch.object_types.get(_guess_type(o), 0) + 1
        sch.support += 1
    return out


def _guess_type(name: str) -> str:
    """Cheap offline type guess (no LLM). Capitalized multiword/orgs -> ORG etc."""
    low = name.lower()
    if any(k in low for k in ("corp", "inc", "ltd", "company", "ai")):
        return "ORG"
    if name[0:1].isupper():
        return "PERSON"
    return "ENTITY"


def apply_schema(schemas: Dict[str, RelationSchema], subject: str, obj: str) -> List[str]:
    """Given induced schemas, predict which relations likely hold for a (subject, obj) pair."""
    st, ot = _guess_type(subject), _guess_type(obj)
    predicted = []
    for rel, sch in schemas.items():
        if sch.confidence() < 0.34:
            continue
        if (st in sch.subject_types and ot in sch.object_types) or sch.support >= 3:
            predicted.append(rel)
    return predicted


async def analogy(source_entity: str, target_entity: str, user_id: str) -> List[Triple]:
    """Structural analogy: copy ``source_entity``'s relation skeleton onto ``target_entity``.

    Returns the predicted new triples for ``target_entity`` (not yet written — the caller
    decides whether to ingest them). This is analogical transfer: "B relates to its
    neighbors the way A relates to A's neighbors."
    """
    async with DatabasePool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT relationship, target_name FROM graph_edges e
            JOIN graph_nodes n ON n.id = e.target_id
            WHERE e.user_id=$1 AND e.source_id = (
                SELECT id FROM graph_nodes WHERE user_id=$1 AND name ILIKE $2 LIMIT 1
            ) AND e.valid_to IS NULL;
            """,
            user_id, source_entity,
        )
        results = []
        for r in rows:
            results.append((target_entity, r["relationship"], r["target_name"]))
        return results


__all__ = ["RelationSchema", "induce_schema", "apply_schema", "analogy", "Triple"]
