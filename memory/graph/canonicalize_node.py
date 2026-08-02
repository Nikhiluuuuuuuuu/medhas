"""Layer 4 (Graphiti): Dynamic Entity Canonicalization & Resolution Module."""

import re
from typing import Optional
from infrastructure.db import DatabasePool
from utils import measure_latency, log_graph, log_error

def normalize_entity_name(name: str) -> str:
    """Dynamically clean and normalize entity strings (Graphiti-style).

    - Strips common corporate suffixes (Inc, LLC, Corp, ...).
    - Collapses runs of internal whitespace to a single space.
    Preserves readability (no space removal) — space-insensitive matching is
    handled separately in ``resolve_canonical_node_name``.
    """
    s = name.strip()
    s = re.sub(r'\s+', ' ', s)  # collapse multiple spaces
    s_clean = re.sub(r'(?i)\b(inc\.?|llc\.?|corp\.?|corporation|ltd\.?|limited)\b', '', s).strip()
    return s_clean if len(s_clean) >= 2 else s


def _space_insensitive(value: str) -> str:
    """Lowercase and remove all whitespace — used for entity-resolution matching."""
    return re.sub(r'\s+', '', value.lower())


async def resolve_canonical_node_name(user_id: str, name: str) -> str:
    """Graphiti-style Entity Resolution: dynamically match entity against existing graph nodes to prevent entity duplicate drift.

    Matching is case- AND space-insensitive (e.g. ``"New York"`` resolves to an
    existing ``"NewYork"`` node), fixing prior fragmentation of the same entity
    written with/without spaces.
    """
    async with measure_latency(f"memory.graph.resolve_canonical_node_name ({name})"):
        normalized = normalize_entity_name(name)
        norm_spaceless = _space_insensitive(normalized)
        try:
            async with DatabasePool.acquire() as conn:
                # 1. Exact (case/space-insensitive) match
                row = await conn.fetchrow(
                    """
                    SELECT name FROM graph_nodes
                    WHERE user_id = $1
                      AND replace(lower(name), ' ', '') = $2
                    LIMIT 1;
                    """,
                    user_id,
                    norm_spaceless
                )
                if row:
                    return row["name"]

                # 2. Prefix / contains match (space-insensitive)
                candidate_rows = await conn.fetch(
                    """
                    SELECT name FROM graph_nodes
                    WHERE user_id = $1 AND (
                        replace(lower(name), ' ', '') LIKE replace(lower($2), ' ', '') || '%'
                        OR replace(lower($2), ' ', '') LIKE replace(lower(name), ' ', '') || '%'
                    )
                    LIMIT 3;
                    """,
                    user_id,
                    norm_spaceless
                )
                if candidate_rows:
                    best_match = candidate_rows[0]["name"]
                    log_graph(f"Canonicalized entity [bold white]'{name}'[/bold white] -> [bold white]'{best_match}'[/bold white]")
                    return best_match

                # 3. Levenshtein distance check if fuzzystrmatch extension is enabled
                try:
                    fuzzy_row = await conn.fetchrow(
                        """
                        SELECT name FROM graph_nodes
                        WHERE user_id = $1
                          AND levenshtein(replace(lower(name), ' ', ''), $2) <= 2
                        ORDER BY levenshtein(replace(lower(name), ' ', ''), $2) ASC
                        LIMIT 1;
                        """,
                        user_id,
                        norm_spaceless
                    )
                    if fuzzy_row:
                        log_graph(f"Canonicalized entity [bold white]'{name}'[/bold white] -> [bold white]'{fuzzy_row['name']}'[/bold white]")
                        return fuzzy_row["name"]
                except Exception:
                    pass

                return normalized
        except Exception:
            return normalized
