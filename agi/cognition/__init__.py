"""Cognition orchestrator — wires perception, reasoning, generalization, embodiment.

A single ``cognitive_step`` takes a raw percept (or text) and produces:
  1. a normalized Percept (perception),
  2. inferred/derived facts via forward-chaining over the agent's graph rules (reasoning),
  3. schema induction + analogical predictions for novel entities (generalization),
  4. an optional action via the agent's body model (embodiment).

It is the thin loop that turns the memory engine into a *cognitive* agent. Every stage
is offline-safe (deterministic; the LLM is only used opportunistically by the perception
extractor's online fallback). The orchestrator never blocks on the network.

Entry points:
  * ``cognitive_step(text, user_id)`` — full pipeline, returns a structured result.
  * ``engine.think(text, user_id)`` — convenience wrapper on MemoryEngine (see agi/engine.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agi.cognition import perception as _perception
from agi.cognition import reasoning as _reasoning
from agi.cognition import generalization as _generalization
from agi.cognition.embodiment import BodyModel, ensure_body_schema


@dataclass
class CognitiveResult:
    percept: Any = None
    derived_facts: List[tuple] = field(default_factory=list)
    schema_predictions: List[str] = field(default_factory=list)
    analogies: List[tuple] = field(default_factory=list)
    action: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)


def _graph_triples_from_edges(edges: List[Dict[str, Any]], source_name: Optional[str] = None) -> List[tuple]:
    out = []
    for e in edges:
        s = e.get("source_name") or source_name
        t = e.get("target_name")
        r = e.get("relationship")
        if s and t and r:
            out.append((s, r, t))
    return out


async def cognitive_step(
    text: str,
    user_id: str,
    *,
    modality: str = "text",
    body: Optional[BodyModel] = None,
    action: Optional[str] = None,
    action_params: Optional[Dict[str, Any]] = None,
) -> CognitiveResult:
    """Run the full cognition pipeline on one input and return a structured result."""
    res = CognitiveResult()

    # 1. PERCEPTION — normalize input into a perceptual symbol.
    percept = await _perception.perceive(text, modality=modality, user_id=user_id)
    res.percept = percept
    res.notes.append(f"perceived({modality}) salience={percept.salience} scene={percept.scene_type}")

    # 2. REASONING — forward-chain over the agent's stored graph + default commonsense rules.
    edges = []
    try:
        from memory.graph.query_subgraph import query_subgraph
        for ent in percept.entities[:5]:
            sg = await query_subgraph(user_id, ent)
            if sg:
                edges.extend(sg.outgoing_edges)
    except Exception as e:
        res.notes.append(f"reasoning graph read skipped: {e}")
    triples = _graph_triples_from_edges(edges)
    # Include the freshly perceived triples as known facts too.
    triples += [(s, r, o) for s, r, o in percept.relations]
    rules = _reasoning.default_rules()
    closure = _reasoning.forward_chain(triples, rules)
    res.derived_facts = [t for t in closure if t not in triples]
    res.notes.append(f"reasoning derived {len(res.derived_facts)} new fact(s) via forward chaining")

    # 3. GENERALIZATION — induce schema from perceived + stored triples and predict relations.
    try:
        schemas = _generalization.induce_schema(triples)
        for s, r, o in percept.relations:
            preds = _generalization.apply_schema(schemas, s, o)
            res.schema_predictions.extend(preds)
        res.schema_predictions = list(dict.fromkeys(res.schema_predictions))
    except Exception as e:
        res.notes.append(f"generalization skipped: {e}")

    # 4. EMBODIMENT — if an action is requested and the body can do it, act + observe.
    if body is not None and action:
        res.action = await body.act(action, text, action_params or {})
        res.notes.append(f"embodiment acted: {action} -> {res.action.get('success')}")

    return res


__all__ = ["cognitive_step", "CognitiveResult", "BodyModel", "ensure_body_schema"]
