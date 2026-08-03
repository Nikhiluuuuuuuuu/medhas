"""2. ASYNC PATH PIPELINE: Non-blocking background Fact & Temporal Graph Extraction.

Production Cognee/Graphiti-style ingest:
  - Anchor each turn as an Episode (Cognee/Graphiti episode model).
  - Feed last-K conversation context into extraction (Mem0 Phase 0 context gathering).
  - Route facts through insert_fact (Mem0 hash dedup + LLM decision matrix).
  - Route edges through upsert_node (space-insensitive canonicalization + semantic merge)
    and update_bayesian_belief (dynamic belief revision).
Supervised fire-and-forget: exceptions are logged, not silently dropped.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from uuid import UUID

import memory.atomic as atomic_mem
import memory.graph as graph_mem
import memory.session as session_mem
from memory.atomic.hashing import content_hash
from infrastructure.llm import GroqLLMProvider
from utils import measure_latency, log_atomic, log_graph, log_error
from pipeline.prompts import EXTRACTION_PROMPT, CONTEXT_TEMPLATE
from config import settings

extractor_llm = GroqLLMProvider()


async def _last_k_context(user_id: str, session_id: Optional[UUID], k: int = 6) -> str:
    try:
        transcript = await session_mem.get_transcript(session_id, limit=k + 1)
        lines = [f"{m.role}: {m.content}" for m in transcript if m.role in ("user", "assistant")]
        return "\n".join(lines[-k:]) if lines else "(no prior context)"
    except Exception:
        return "(no prior context)"


async def extract_and_persist_background(
    user_id: str,
    user_message: str,
    assistant_response: str,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None
) -> None:
    """Async Cognee/Graphiti background task: anchor episode, extract, route to memory layers."""
    async with measure_latency("pipeline.async_extractor.extract_and_persist_background"):
        try:
            # 0. Anchor episode (Cognee/Graphiti episode model)
            from infrastructure.db import DatabasePool
            async with DatabasePool.acquire() as conn:
                ep = await conn.fetchrow(
                    """
                    INSERT INTO episodes (user_id, session_id, agent_id, content, source)
                    VALUES ($1, $2, $3, $4, 'message')
                    RETURNING id;
                    """,
                    user_id, session_id, agent_id,
                    f"User: {user_message}\nAssistant: {assistant_response}",
                )
                episode_id = ep["id"] if ep else None

            # 1. Build context-aware extraction prompt (Mem0 Phase 0)
            context = "(no prior context)"
            if session_id is not None:
                context = await _last_k_context(user_id, session_id, k=settings.MAX_HISTORICAL_MESSAGES)
            context_block = CONTEXT_TEMPLATE.format(k=settings.MAX_HISTORICAL_MESSAGES, context=context)
            messages = [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"{context_block}\nUser turn: '{user_message}'\nAssistant response: '{assistant_response}'"},
            ]
            response = await extractor_llm.chat_completion(messages, temperature=0.0)
            raw_content = (response.get("content", "") or "").strip()
            if not raw_content:
                return
            if "```json" in raw_content:
                raw_content = raw_content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```", 1)[1].split("```", 1)[0].strip()

            data = json.loads(raw_content)

            # 2. Persist facts (Mem0: hash dedup + LLM decision matrix inside insert_fact)
            # Mem0 main.py:985 — within-batch hash dedup so the same fact text extracted
            # twice in one LLM response is only stored once (not just across calls).
            facts: List[str] = data.get("facts", [])
            seen_hashes = set()
            for fact_text in facts:
                fact_hash = content_hash(fact_text)
                if fact_hash in seen_hashes:
                    continue
                seen_hashes.add(fact_hash)
                await atomic_mem.insert_fact(user_id, fact_text, session_id=session_id, agent_id=agent_id)

            # 3. Persist bi-temporal graph edges (Graphiti: canonicalization + semantic merge)
            # Cognee deduplicate_nodes_and_edges.py — within-batch edge dedup keyed by
            # (source, target, relationship) so identical edges in one batch merge once.
            edges: List[Dict[str, Any]] = data.get("edges", [])
            seen_edge_keys = set()
            for edge_info in edges:
                src_name = edge_info.get("source")
                tgt_name = edge_info.get("target")
                if not src_name or not tgt_name:
                    continue
                rel = edge_info.get("relationship", "associated_with")
                edge_key = (src_name, tgt_name, rel)
                if edge_key in seen_edge_keys:
                    continue
                seen_edge_keys.add(edge_key)

                src_node = await graph_mem.upsert_node(
                    user_id, name=src_name, entity_type=edge_info.get("source_type", "Person"),
                    session_id=session_id, agent_id=agent_id,
                )
                tgt_node = await graph_mem.upsert_node(
                    user_id, name=tgt_name, entity_type=edge_info.get("target_type", "Entity"),
                    session_id=session_id, agent_id=agent_id,
                )

                conf = float(edge_info.get("confidence", 0.85))
                conf = max(0.50, min(0.99, conf))
                await graph_mem.update_bayesian_belief(user_id, tgt_node.name, likelihood_evidence=conf)

                ts_str = edge_info.get("timestamp")
                valid_from = datetime.now(timezone.utc)
                if ts_str:
                    try:
                        valid_from = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                    except Exception:
                        valid_from = datetime.now(timezone.utc)

                await graph_mem.update_edge(
                    user_id=user_id, source_id=src_node.id, target_id=tgt_node.id,
                    relationship=rel,
                    valid_from=valid_from, session_id=session_id, agent_id=agent_id,
                )

            if facts or edges:
                log_graph(
                    f"Background extraction persisted {len(facts)} facts and {len(edges)} graph edges "
                    f"(episode={episode_id})"
                )

        except Exception as e:
            log_error(f"Background extraction handled error: {e}")
