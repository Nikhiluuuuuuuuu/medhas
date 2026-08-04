"""E2/E3/E7/E8 — Consolidation engine.

E2 Episodic → semantic compression: old raw episodes are distilled into a durable
   gist fact, then marked compressed (the "sleep" transformation, not just summary).
E3 Procedural auto-induction: repeated successful action sequences become skills.
E7 Consolidation scheduler: idle/periodic trigger that runs all consolidation phases.
E8 A-MEM style memory evolution: new memories retro-link and revise their neighbours
   instead of being appended in isolation (Zettelkasten link + context update).
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from infrastructure.db import DatabasePool
from infrastructure.llm import GroqLLMProvider
from utils import log_atomic, log_error, measure_latency

llm = GroqLLMProvider()

GIST_PROMPT = """You compress raw conversation episodes into durable semantic gist facts.
Drop chit-chat and transient detail. Keep stable, reusable knowledge about the user/world.
Return ONLY JSON: {"gists": ["fact 1", "fact 2"]}
Return {"gists": []} if nothing durable is present."""

SKILL_PROMPT = """You detect repeated PROCEDURES in an agent's action history.
If the same multi-step sequence solved the same kind of task 2+ times, emit it as a skill.
Return ONLY JSON: {"skills": [{"task": "short task pattern", "steps": ["step 1", "step 2"]}]}
Return {"skills": []} if no repeated procedure exists."""

EVOLVE_PROMPT = """You maintain a Zettelkasten memory network (A-MEM).
Given a NEW memory and its NEAREST NEIGHBOURS, decide how the network should evolve.
Return ONLY JSON:
{"links": ["<neighbour_id>"], "revisions": [{"id": "<neighbour_id>", "new_text": "revised text"}]}
Only revise a neighbour when the new memory genuinely refines or corrects it. Prefer links over revisions."""

EPISODE_AGE_DAYS = 3
MIN_EPISODES_TO_COMPRESS = 3


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}


# --------------------------------------------- E2 episodic -> semantic gist

async def compress_episodes(user_id: str, older_than_days: int = EPISODE_AGE_DAYS) -> Dict[str, Any]:
    """Distil aged raw episodes into durable semantic gist facts, then mark compressed."""
    from memory.atomic import insert_fact

    async with measure_latency("agi.consolidation.compress_episodes"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, content, reference_time
                    FROM episodes
                    WHERE user_id = $1 AND compressed = FALSE
                      AND reference_time < NOW() - ($2 || ' days')::interval
                    ORDER BY reference_time ASC
                    LIMIT 25;
                    """,
                    user_id, str(older_than_days),
                )
            if len(rows) < MIN_EPISODES_TO_COMPRESS:
                return {"status": "skipped", "reason": "not enough aged episodes",
                        "episodes": len(rows), "gists": 0}

            blob = "\n\n".join(f"[{r['reference_time']}] {r['content'][:600]}" for r in rows)
            resp = await llm.chat_completion(
                [{"role": "system", "content": GIST_PROMPT},
                 {"role": "user", "content": f"Episodes:\n{blob}"}],
                temperature=0.1,
            )
            gists = _parse_json(resp.get("content", "")).get("gists", []) or []

            gist_ids: List[str] = []
            for g in gists[:10]:
                text = str(g).strip()
                if not text:
                    continue
                fact = await insert_fact(
                    user_id, f"[Gist] {text}", memory_type="semantic",
                    metadata={"source": "episodic_compression"},
                )
                gist_ids.append(str(fact.id))

            async with DatabasePool.acquire() as conn:
                await conn.execute(
                    "UPDATE episodes SET compressed = TRUE WHERE id = ANY($1::uuid[]);",
                    [r["id"] for r in rows],
                )
                if gist_ids:
                    await conn.execute(
                        "UPDATE episodes SET gist_fact_id = $2 WHERE id = ANY($1::uuid[]);",
                        [r["id"] for r in rows], UUID(gist_ids[0]),
                    )
            log_atomic(f"E2 compressed {len(rows)} episodes -> {len(gist_ids)} gists")
            return {"status": "success", "episodes": len(rows), "gists": len(gist_ids)}
        except Exception as e:
            log_error(f"compress_episodes failed: {e}")
            return {"status": "error", "message": str(e), "gists": 0}


# ------------------------------------------- E3 procedural skill induction

async def induce_skills(user_id: str) -> Dict[str, Any]:
    """Mine repeated successful action sequences from tool history into skill playbooks."""
    from memory.procedural import store_skill_playbook

    async with measure_latency("agi.consolidation.induce_skills"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT m.content, m.created_at
                    FROM messages m
                    JOIN sessions s ON s.id = m.session_id
                    WHERE s.user_id = $1 AND m.role = 'tool'
                    ORDER BY m.created_at DESC
                    LIMIT 60;
                    """,
                    user_id,
                )
            if len(rows) < 4:
                return {"status": "skipped", "reason": "insufficient action history", "skills": 0}

            history = "\n".join(f"- {r['content'][:220]}" for r in reversed(rows))
            resp = await llm.chat_completion(
                [{"role": "system", "content": SKILL_PROMPT},
                 {"role": "user", "content": f"Action history:\n{history}"}],
                temperature=0.1,
            )
            skills = _parse_json(resp.get("content", "")).get("skills", []) or []
            stored = 0
            for s in skills[:5]:
                task = str(s.get("task", "")).strip()
                steps = [str(x) for x in (s.get("steps") or []) if str(x).strip()]
                if task and len(steps) >= 2:
                    await store_skill_playbook(user_id, task, steps)
                    stored += 1
            if stored:
                log_atomic(f"E3 induced {stored} procedural skills")
            return {"status": "success", "skills": stored}
        except Exception as e:
            log_error(f"induce_skills failed: {e}")
            return {"status": "error", "message": str(e), "skills": 0}


# ------------------------------------------------- E8 A-MEM memory evolution

async def evolve_memory_network(
    user_id: str,
    new_fact_id: UUID,
    new_fact_text: str,
    neighbours: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """A-MEM: retro-link the new memory to neighbours and revise those it refines."""
    async with measure_latency("agi.consolidation.evolve_memory_network"):
        try:
            if not neighbours:
                from memory.atomic import search_facts
                neighbours = await search_facts(user_id, new_fact_text, limit=5)
            neighbours = [n for n in (neighbours or []) if str(getattr(n, "id", "")) != str(new_fact_id)]
            if not neighbours:
                return {"links": 0, "revisions": 0}

            listing = "\n".join(f"{n.id}: {n.fact_text}" for n in neighbours)
            resp = await llm.chat_completion(
                [{"role": "system", "content": EVOLVE_PROMPT},
                 {"role": "user", "content": f"NEW MEMORY: {new_fact_text}\n\nNEIGHBOURS:\n{listing}"}],
                temperature=0.1,
            )
            data = _parse_json(resp.get("content", ""))
            valid_ids = {str(n.id) for n in neighbours}

            link_ids: List[UUID] = []
            for lid in (data.get("links") or []):
                if str(lid) in valid_ids:
                    try:
                        link_ids.append(UUID(str(lid)))
                    except Exception:
                        pass

            revisions = 0
            async with DatabasePool.acquire() as conn:
                if link_ids:
                    await conn.execute(
                        """
                        UPDATE atomic_facts
                        SET linked_ids = (
                            SELECT ARRAY(SELECT DISTINCT unnest(COALESCE(linked_ids, ARRAY[]::uuid[]) || $2::uuid[]))
                        )
                        WHERE id = $1;
                        """,
                        new_fact_id, link_ids,
                    )
                    # bidirectional backlink
                    await conn.execute(
                        """
                        UPDATE atomic_facts
                        SET linked_ids = (
                            SELECT ARRAY(SELECT DISTINCT unnest(COALESCE(linked_ids, ARRAY[]::uuid[]) || ARRAY[$2]::uuid[]))
                        )
                        WHERE id = ANY($1::uuid[]);
                        """,
                        link_ids, new_fact_id,
                    )
                for rev in (data.get("revisions") or [])[:5]:
                    rid, new_text = str(rev.get("id", "")), str(rev.get("new_text", "")).strip()
                    if rid in valid_ids and new_text:
                        protected = await conn.fetchval(
                            "SELECT is_protected FROM atomic_facts WHERE id=$1;", UUID(rid))
                        if protected:
                            continue  # E33: protected core is never rewritten
                        await conn.execute(
                            "UPDATE atomic_facts SET fact_text = $2, note = COALESCE(note,'') || ' [A-MEM revised]' WHERE id = $1;",
                            UUID(rid), new_text,
                        )
                        revisions += 1
            if link_ids or revisions:
                log_atomic(f"E8 A-MEM evolution: links={len(link_ids)} revisions={revisions}")
            return {"links": len(link_ids), "revisions": revisions}
        except Exception as e:
            log_error(f"evolve_memory_network failed: {e}")
            return {"links": 0, "revisions": 0, "error": str(e)}


# ---------------------------------------------------- E7 consolidation cycle

async def run_consolidation(user_id: str, include_dream: bool = True) -> Dict[str, Any]:
    """Full offline consolidation pass (E7): the scheduler's unit of work."""
    from agi.forgetting import protect_core_memories, run_forgetting_sweep
    from agi.sensory import promote_percepts, sweep_expired

    async with measure_latency("agi.consolidation.run_consolidation"):
        out: Dict[str, Any] = {"user_id": user_id}
        out["sensory"] = await promote_percepts(user_id)
        out["sensory_expired"] = await sweep_expired()
        out["compression"] = await compress_episodes(user_id)
        out["skills"] = await induce_skills(user_id)
        out["protected"] = await protect_core_memories(user_id)
        out["forgetting"] = await run_forgetting_sweep(user_id)
        if include_dream:
            try:
                from memory.atomic import run_dream_cycle
                out["dream"] = await run_dream_cycle(user_id)
            except Exception as e:
                out["dream"] = {"status": "error", "message": str(e)}
        log_atomic(f"E7 consolidation complete for {user_id}")
        return out
