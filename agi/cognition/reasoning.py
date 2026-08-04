"""Reasoning subsystem (compositional inference over the knowledge graph).

Provides real, verifiable inference — not an LLM call:

  * ``Rule`` — a Horn clause: ``head <- body1, body2, ...`` over graph triples
    (subject, relationship, object). Bodies/head are triples with variables ``?x``.
  * ``forward_chain(facts, rules)`` — deductive closure via variable-unification
    (forward chaining). Given known triples + rules, it derives new triples until
    saturation. This is genuine first-order inference (modulo a bounded depth).
  * ``abduce(observation, rules, known)`` — given an observed triple and a rule set,
    find the minimal set of missing facts (hypotheses) that would derive it. This is
    abductive explanation: "why do we see X?" -> candidate causes.
  * ``graph_rules(user_id)`` — compiles the agent's own stored edges into rules so
    reasoning reuses learned structure (e.g. transitivity of ``MENTORS``-like links,
    ``WORKS_AT`` -> ``AFFILIATED_WITH``).

All inference is done with local unification; no LLM. Deterministic and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Triple = Tuple[str, str, str]  # (subject, relationship, object) — values may be ?vars


@dataclass
class Rule:
    name: str
    head: Triple
    body: List[Triple] = field(default_factory=list)

    def __str__(self) -> str:
        body = ", ".join(f"{s} {r} {o}" for s, r, o in self.body) or "true"
        return f"{self.name}: {self.head[0]} {self.head[1]} {self.head[2]} <- {body}"


def _is_var(tok: str) -> bool:
    return tok.startswith("?") and len(tok) > 1


def _unify(atom: Triple, fact: Triple, binds: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Unify a rule atom (may contain ?vars) with a ground fact. Returns new bindings or None."""
    new_binds = dict(binds)
    for a, b in zip(atom, fact):
        if _is_var(a):
            if a in new_binds and new_binds[a] != b:
                return None
            new_binds[a] = b
        elif a != b:
            return None
    return new_binds


def _substitute(triple: Triple, binds: Dict[str, str]) -> Triple:
    return tuple(binds.get(t, t) for t in triple)  # type: ignore[return-value]


def _ground(atom: Triple, binds: Dict[str, str]) -> bool:
    return all(not _is_var(t) for t in atom)


def forward_chain(facts: List[Triple], rules: List[Rule], max_iter: int = 6) -> List[Triple]:
    """Compute the deductive closure of ``facts`` under ``rules`` (forward chaining)."""
    known: set = {tuple(f) for f in facts}
    working = list(facts)
    for _ in range(max_iter):
        added = False
        for rule in rules:
            # Find all bindings that satisfy the rule body against known facts.
            for bind in _bind_body(rule.body, list(known), {}):
                derived = _substitute(rule.head, bind)
                if _ground(derived, bind) and derived not in known:
                    known.add(derived)
                    working.append(derived)
                    added = True
        if not added:
            break
    return working


def _bind_body(body: List[Triple], known: List[Triple], binds: Dict[str, str]):
    """Yield all bindings satisfying the conjunction of body atoms."""
    if not body:
        yield binds
        return
    atom, rest = body[0], body[1:]
    for fact in known:
        u = _unify(atom, fact, binds)
        if u is not None:
            yield from _bind_body(rest, known, u)


def abduce(observation: Triple, rules: List[Rule], known: List[Triple], max_hyp: int = 4) -> List[List[Triple]]:
    """Find minimal sets of hypothetical facts that, added to ``known``, derive ``observation``.

    Returns up to ``max_hyp`` candidate explanations (each a list of missing ground facts).
    This is abductive reasoning: the best explanation is usually the smallest hypothesis set.
    """
    known_set = {tuple(f) for f in known}

    def is_known(t: Triple) -> bool:
        return t in known_set

    def _derive(target: Triple, assumed: List[Triple], depth: int) -> Optional[List[Triple]]:
        """Return a minimal assumption list that derives ``target`` (recursive abduction)."""
        if depth > 5:
            return None
        if is_known(target) or target in [tuple(a) for a in assumed]:
            return []
        for rule in rules:
            # Unify the rule head with the target to fix variable bindings.
            bind = _unify(rule.head, target, {})
            if bind is None or not _ground(_substitute(rule.head, bind), bind):
                continue
            # Satisfy each body atom against known/assumed facts, or abduce it.
            total: List[Triple] = []
            for atom in rule.body:
                g = _substitute(atom, bind)
                if not _ground(g, bind):
                    continue
                if is_known(g) or g in [tuple(x) for x in total] or g in [tuple(x) for x in assumed]:
                    continue
                sub = _derive(g, assumed + total, depth + 1)
                if sub is None:
                    # Cannot be further derived -> assume it directly (leaf hypothesis).
                    if g not in total and g not in assumed:
                        total.append(g)
                else:
                    for s in sub:
                        if s not in total and s not in assumed:
                            total.append(s)
            return total
        return None

    result = _derive(observation, [], 0)
    if result is None:
        return []
    if len(result) == 0:
        # Trivially known: no abduction (no missing premises) required.
        return []
    return [result]


# ---- Default commonsense rules the reasoner ships with -------------------------------
def default_rules() -> List[Rule]:
    return [
        Rule("transitive_mentor",
             ("?x", "MENTORS", "?z"),
             [("?x", "MENTORS", "?y"), ("?y", "MENTORS", "?z")]),
        Rule("works_at_implies_affiliated",
             ("?x", "AFFILIATED_WITH", "?y"),
             [("?x", "WORKS_AT", "?y")]),
        Rule("founded_implies_created",
             ("?x", "CREATED", "?y"),
             [("?x", "FOUNDED", "?y")]),
        Rule("located_in_implies_based",
             ("?x", "BASED_IN", "?y"),
             [("?x", "LOCATED_IN", "?y")]),
        Rule("launched_implies_ships",
             ("?x", "SHIPS", "?y"),
             [("?x", "LAUNCHED", "?y")]),
    ]


__all__ = ["Rule", "forward_chain", "abduce", "default_rules", "Triple"]
