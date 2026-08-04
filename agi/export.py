"""E26 — Backup, export & portability.

Snapshot a user's entire memory footprint (facts, episodes, edges, profile, skills,
intentions, meta-memory, percepts) into a single portable JSON bundle, and a restore
path. This is the durability/portability requirement for a long-horizon memory store.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from infrastructure.db import DatabasePool
from utils import log_error, log_atomic


async def export_user_memory(user_id: str) -> Dict[str, Any]:
    """Produce a self-contained JSON snapshot of a user's memory (E26)."""
    bundle: Dict[str, Any] = {
        "schema_version": 1,
        "user_id": user_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "tables": {},
    }
    queries = {
        "atomic_facts": "SELECT * FROM atomic_facts WHERE user_id=$1",
        "episodes": "SELECT * FROM episodes WHERE user_id=$1",
        "graph_edges": "SELECT * FROM graph_edges WHERE user_id=$1",
        "graph_nodes": "SELECT * FROM graph_nodes WHERE user_id=$1",
        "prospective_memory": "SELECT * FROM prospective_memory WHERE user_id=$1",
        "meta_memory": "SELECT * FROM meta_memory WHERE user_id=$1",
        "user_profile": "SELECT * FROM user_profile WHERE user_id=$1",
        "percept_buffer": "SELECT * FROM percept_buffer WHERE user_id=$1",
    }
    try:
        async with DatabasePool.acquire() as conn:
            for name, q in queries.items():
                rows = await conn.fetch(q, user_id)
                bundle["tables"][name] = [dict(r) for r in rows]
            # procedural playbooks from atomic_facts (memory_type procedural)
            rows = await conn.fetch(
                "SELECT id, fact_text, metadata FROM atomic_facts WHERE user_id=$1 AND memory_type='procedural';",
                user_id,
            )
            bundle["tables"]["procedural_playbooks"] = [dict(r) for r in rows]
        log_atomic(f"E26 exported memory for {user_id} ({sum(len(v) for v in bundle['tables'].values())} rows)")
        return bundle
    except Exception as e:
        log_error(f"export_user_memory failed: {e}")
        return {"error": str(e)}


def export_to_file(bundle: Dict[str, Any], path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, default=str, indent=2)
    return path


async def import_user_memory(bundle: Dict[str, Any], *, overwrite: bool = False) -> Dict[str, Any]:
    """Restore a memory bundle. Inserts rows; safe to run idempotently (ON CONFLICT DO NOTHING)."""
    user_id = bundle.get("user_id")
    if not user_id:
        return {"status": "error", "message": "no user_id in bundle"}
    inserted = 0
    try:
        async with DatabasePool.acquire() as conn:
            async with conn.transaction():
                for table, rows in bundle.get("tables", {}).items():
                    for r in rows:
                        cols = [k for k in r.keys() if k != "id"]
                        if not cols:
                            continue
                        placeholders = ", ".join(f"${i+1}" for i in range(len(cols) + 1))
                        col_list = "id, " + ", ".join(cols)
                        await conn.execute(
                            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                            f"ON CONFLICT (id) DO NOTHING;",
                            r.get("id"), *[r[c] for c in cols],
                        )
                        inserted += 1
        log_atomic(f"E26 imported {inserted} rows for {user_id}")
        return {"status": "imported", "rows": inserted}
    except Exception as e:
        log_error(f"import_user_memory failed: {e}")
        return {"status": "error", "message": str(e), "rows": inserted}
