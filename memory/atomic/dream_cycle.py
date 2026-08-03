import json
from datetime import datetime, timezone
from typing import List, Dict, Any
from infrastructure.db import DatabasePool
from infrastructure.llm import GroqLLMProvider, FastEmbeddingProvider
from memory.atomic.insert_fact import insert_fact
from memory.working import update_block
import memory.graph as graph_mem
from utils import measure_latency, log_atomic, log_graph, log_error

dream_llm = GroqLLMProvider()
embedder = FastEmbeddingProvider()

PATTERNS_PROMPT = """\
You mine recurring behavioral or topical PATTERNS from a set of memory reflections.
Return ONLY a JSON object: {"patterns": ["pattern 1", "pattern 2"]}
Each pattern is a concise, durable insight (e.g. "User consistently prefers local-first tools").
If no clear pattern exists, return {"patterns": []}."""

DREAM_CYCLE_PROMPT = """
You are an advanced multi-layer memory consolidation worker (Dream Cycle).
Analyze recent user facts and synthesize higher-level insights across all memory layers:

1. Synthesize high-level reflections/insights.
2. Update consolidated user profile.
3. Extract active goals & scratchpad context.
4. Extract entity relationships to build/update the knowledge graph (source entity, target entity, relationship type, confidence between 0.50 and 0.99).

Return ONLY a JSON object with this schema:
{
    "reflections": ["high level insight 1", "high level insight 2"],
    "summary_user_profile": "Updated consolidated profile of the user",
    "scratchpad_summary": "Current active focus and context",
    "active_goals": ["Goal 1", "Goal 2"],
    "edges": [
        {
            "source": "User",
            "source_type": "Person",
            "target": "TargetEntity",
            "target_type": "Entity",
            "relationship": "relationship_type",
            "confidence": 0.88
        }
    ]
}
""".strip()

async def run_dream_cycle(user_id: str) -> Dict[str, Any]:
    """Execute Dream Cycle consolidation: reflect on facts, update Letta RAM blocks, atomic vector facts, and Zep Knowledge Graph edges with Bayesian updates."""
    async with measure_latency("memory.atomic.run_dream_cycle"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT fact_text, importance_score
                    FROM atomic_facts
                    WHERE user_id = $1 AND is_active = TRUE AND fact_text NOT LIKE '[Reflection]%'
                    ORDER BY created_at DESC
                    LIMIT 15;
                    """,
                    user_id
                )
                if not rows:
                    return {"status": "no_facts_to_reflect"}

                facts_summary = "\n".join([f"- {r['fact_text']} (importance: {r['importance_score']})" for r in rows])

                messages = [
                    {"role": "system", "content": DREAM_CYCLE_PROMPT},
                    {"role": "user", "content": f"Recent facts:\n{facts_summary}"}
                ]
                response = await dream_llm.chat_completion(messages, temperature=0.2)
                raw = response.get("content", "").strip()

                reflections = []
                edges = []
                try:
                    if "```json" in raw:
                        raw = raw.split("```json")[1].split("```")[0].strip()
                    elif "```" in raw:
                        raw = raw.split("```")[1].split("```")[0].strip()
                    parsed = json.loads(raw)
                    
                    reflections = parsed.get("reflections", [])
                    profile_summary = parsed.get("summary_user_profile", "")
                    scratchpad = parsed.get("scratchpad_summary", "")
                    active_goals = parsed.get("active_goals", [])
                    edges = parsed.get("edges", [])

                    # 1. Layer 2: Update Letta Working Memory RAM blocks
                    if profile_summary:
                        await update_block(user_id, "user_profile", str(profile_summary))
                    if scratchpad:
                        await update_block(user_id, "scratchpad", str(scratchpad))
                    if active_goals:
                        # active_goals is a list per the prompt schema; blocks store strings
                        goals_str = "\n".join(f"- {g}" for g in active_goals) if isinstance(active_goals, list) else str(active_goals)
                        await update_block(user_id, "active_goals", goals_str)

                except Exception as parse_err:
                    log_error(f"Dream cycle JSON parse fallback: {parse_err}")
                    lines = [line.strip("-* \t'\"").strip() for line in raw.split("\n") if line.strip()]
                    reflections = [line for line in lines if len(line) > 10 and not line.startswith(("{", "[", '"reflections"', "'reflections'"))][:3]

                # 2. Layer 3: Insert high-level reflections as active facts
                for r_text in reflections:
                    reflection_fact = f"[Reflection] {r_text}"
                    log_atomic(f"🌙 [DREAM CYCLE REFLECTION] Synthesized: [bold white]'{r_text}'[/bold white]")
                    await insert_fact(user_id, reflection_fact, is_reflection=True)

                # 3. Layer 4: Build Knowledge Graph nodes, edges & Bayesian belief updates from Dream Cycle
                graph_updated_count = 0
                for edge_info in edges:
                    src_name = edge_info.get("source")
                    tgt_name = edge_info.get("target")
                    if not src_name or not tgt_name:
                        continue

                    src_node = await graph_mem.upsert_node(
                        user_id,
                        name=src_name,
                        entity_type=edge_info.get("source_type", "Person")
                    )
                    tgt_node = await graph_mem.upsert_node(
                        user_id,
                        name=tgt_name,
                        entity_type=edge_info.get("target_type", "Entity")
                    )

                    # Dynamic Bayesian belief update on consolidated entity
                    conf = float(edge_info.get("confidence", 0.88))
                    conf = max(0.50, min(0.99, conf))
                    await graph_mem.update_bayesian_belief(user_id, tgt_node.name, likelihood_evidence=conf)

                    await graph_mem.update_edge(
                        user_id=user_id,
                        source_id=src_node.id,
                        target_id=tgt_node.id,
                        relationship=edge_info.get("relationship", "associated_with")
                    )
                    graph_updated_count += 1

                if graph_updated_count > 0:
                    log_graph(f"🌙 [DREAM CYCLE GRAPH] Synthesized [bold green]{graph_updated_count}[/bold green] new knowledge graph connections!")

                # 3. (NEW) Pattern extraction: synthesize cross-session recurring themes from
                # reflections (GBrain `dream` patterns phase). Stores pattern facts for recall.
                patterns = []
                try:
                    reflections_text = "\n".join(f"- {r}" for r in reflections)
                    if reflections_text:
                        pat_msgs = [
                            {"role": "system", "content": PATTERNS_PROMPT},
                            {"role": "user", "content": f"Reflections to mine for patterns:\n{reflections_text}"}
                        ]
                        pat_resp = await dream_llm.chat_completion(pat_msgs, temperature=0.3)
                        pat_raw = (pat_resp.get("content", "") or "").strip()
                        if "```json" in pat_raw:
                            pat_raw = pat_raw.split("```json", 1)[1].split("```", 1)[0].strip()
                        elif "```" in pat_raw:
                            pat_raw = pat_raw.split("```", 1)[1].split("```", 1)[0].strip()
                        try:
                            pat_data = json.loads(pat_raw)
                            for p in pat_data.get("patterns", []):
                                ptext = f"[Pattern] {p}" if not str(p).startswith("[Pattern]") else str(p)
                                await insert_fact(user_id, ptext, is_reflection=True)
                                patterns.append(ptext)
                        except Exception:
                            pass
                except Exception as pe:
                    log_error(f"Dream cycle patterns phase skipped: {pe}")

                # 4. (NEW) Embedding refresh: backfill any atomic facts missing an embedding.
                embed_refreshed = 0
                try:
                    async with DatabasePool.acquire() as conn:
                        missing = await conn.fetch(
                            "SELECT id, fact_text FROM atomic_facts WHERE user_id=$1 AND is_active=TRUE AND embedding IS NULL LIMIT 50;",
                            user_id,
                        )
                        for m in missing:
                            try:
                                emb = await embedder.embed_text(m["fact_text"])
                                vec = f"[{','.join(str(x) for x in emb)}]"
                                await conn.execute("UPDATE atomic_facts SET embedding=$2::vector WHERE id=$1", m["id"], vec)
                                embed_refreshed += 1
                            except Exception:
                                pass
                except Exception as ee:
                    log_error(f"Dream cycle embed-refresh skipped: {ee}")

                # 5. (NEW) Orphan detection: active facts never retrieved/referenced recently.
                orphans = 0
                try:
                    async with DatabasePool.acquire() as conn:
                        orphans = await conn.fetchval(
                            """
                            SELECT count(*) FROM atomic_facts
                            WHERE user_id=$1 AND is_active=TRUE
                              AND created_at < NOW() - INTERVAL '30 days'
                              AND importance_score <= 2.0;
                            """,
                            user_id,
                        ) or 0
                except Exception as oe:
                    log_error(f"Dream cycle orphan detection skipped: {oe}")

                log_atomic(f"🌙 [DREAM CYCLE] reflections={len(reflections)} patterns={len(patterns)} "
                           f"embed_refreshed={embed_refreshed} orphans={orphans}")

                return {
                    "status": "success",
                    "reflections": reflections,
                    "patterns": patterns,
                    "graph_edges_created": graph_updated_count,
                    "embed_refreshed": embed_refreshed,
                    "orphans_detected": int(orphans),
                }

        except Exception as e:
            log_error(f"Dream cycle consolidation error: {e}")
            return {"status": "error", "message": str(e)}

