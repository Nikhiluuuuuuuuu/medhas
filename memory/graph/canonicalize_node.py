"""Layer 4 (Graphiti): Dynamic Entity Canonicalization & Resolution Module."""

import re
from typing import Optional
from infrastructure.db import DatabasePool
from utils import measure_latency, log_graph, log_error

def normalize_entity_name(name: str) -> str:
    """Dynamically clean and normalize entity strings."""
    s = name.strip()
    # Remove common corporate suffixes dynamically
    s_clean = re.sub(r'(?i)\b(inc\.?|llc\.?|corp\.?|corporation|ltd\.?|limited)\b', '', s).strip()
    return s_clean if len(s_clean) >= 2 else s

async def resolve_canonical_node_name(user_id: str, name: str) -> str:
    """Graphiti-style Entity Resolution: Dynamically match entity against existing graph nodes to prevent entity duplicate drift."""
    async with measure_latency(f"memory.graph.resolve_canonical_node_name ({name})"):
        normalized = normalize_entity_name(name)
        try:
            async with DatabasePool.acquire() as conn:
                # 1. Exact match on normalized or original name (case-insensitive)
                row = await conn.fetchrow(
                    """
                    SELECT name FROM graph_nodes
                    WHERE user_id = $1 AND (LOWER(name) = LOWER($2) OR LOWER(name) = LOWER($3))
                    LIMIT 1;
                    """,
                    user_id,
                    name,
                    normalized
                )
                if row:
                    return row["name"]
                
                # 2. Dynamic prefix/contains & Levenshtein check for close candidate entities
                candidate_rows = await conn.fetch(
                    """
                    SELECT name FROM graph_nodes
                    WHERE user_id = $1 AND (
                        LOWER(name) LIKE LOWER($2) || '%' 
                        OR LOWER($2) LIKE LOWER(name) || '%'
                    )
                    LIMIT 3;
                    """,
                    user_id,
                    normalized
                )
                if candidate_rows:
                    best_match = candidate_rows[0]["name"]
                    log_graph(f"Canonicalized entity [bold white]'{name}'[/bold white] -> [bold white]'{best_match}'[/bold white]")
                    return best_match

                # 3. Dynamic Levenshtein distance check if fuzzystrmatch extension is enabled
                try:
                    fuzzy_row = await conn.fetchrow(
                        """
                        SELECT name FROM graph_nodes
                        WHERE user_id = $1 AND levenshtein(LOWER(name), LOWER($2)) <= 2
                        ORDER BY levenshtein(LOWER(name), LOWER($2)) ASC
                        LIMIT 1;
                        """,
                        user_id,
                        normalized
                    )
                    if fuzzy_row:
                        log_graph(f"Canonicalized entity [bold white]'{name}'[/bold white] -> [bold white]'{fuzzy_row['name']}'[/bold white]")
                        return fuzzy_row["name"]
                except Exception:
                    pass

                return normalized
        except Exception:
            return normalized
