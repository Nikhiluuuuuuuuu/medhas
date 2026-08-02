"""2. ASYNC PATH PIPELINE: Non-blocking Background Fact & Temporal Graph Extraction via LLM."""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List
import memory.atomic as atomic_mem
import memory.graph as graph_mem
from infrastructure.llm import GroqLLMProvider
from utils import measure_latency, log_atomic, log_graph, log_error

extractor_llm = GroqLLMProvider()

EXTRACTION_PROMPT = """
You are a memory extraction worker. Analyze the user's message and extract:
1. New atomic facts or preference updates.
2. Entity relationship shifts (source entity, target entity, relationship type, timestamp, evidence confidence between 0.50 and 0.99).

Return ONLY a JSON object with this exact schema:
{
    "facts": ["extracted fact 1"],
    "edges": [
        {
            "source": "User",
            "source_type": "Person",
            "target": "TargetEntity",
            "target_type": "Entity",
            "relationship": "relationship_type",
            "timestamp": "2024-01-01T00:00:00Z",
            "confidence": 0.85
        }
    ]
}
If no new facts or edges are present, return {"facts": [], "edges": []}.
""".strip()

from typing import Dict, Any, List, Optional
from uuid import UUID

async def extract_and_persist_background(
    user_id: str,
    user_message: str,
    assistant_response: str,
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None
) -> None:
    """Async Cognee ECL background task parsing conversation turn for facts & temporal graph edges via LLM."""
    async with measure_latency("pipeline.async_extractor.extract_and_persist_background"):
        try:
            messages = [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"User turn: '{user_message}'\nAssistant response: '{assistant_response}'"}
            ]
            response = await extractor_llm.chat_completion(messages, temperature=0.0)
            raw_content = response.get("content", "").strip()

            if not raw_content:
                return

            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(raw_content)

            # Persist facts extracted by LLM
            facts: List[str] = data.get("facts", [])
            for fact_text in facts:
                await atomic_mem.insert_fact(user_id, fact_text, session_id=session_id, agent_id=agent_id)

            # Persist bi-temporal graph edges extracted by LLM
            edges: List[Dict[str, Any]] = data.get("edges", [])
            for edge_info in edges:
                src_name = edge_info.get("source")
                tgt_name = edge_info.get("target")
                if not src_name or not tgt_name:
                    continue

                src_node = await graph_mem.upsert_node(
                    user_id,
                    name=src_name,
                    entity_type=edge_info.get("source_type", "Person"),
                    session_id=session_id,
                    agent_id=agent_id
                )
                tgt_node = await graph_mem.upsert_node(
                    user_id,
                    name=tgt_name,
                    entity_type=edge_info.get("target_type", "Entity"),
                    session_id=session_id,
                    agent_id=agent_id
                )
                
                # Dynamic Bayesian belief revision update driven by evidence confidence
                conf = float(edge_info.get("confidence", 0.85))
                conf = max(0.50, min(0.99, conf))
                await graph_mem.update_bayesian_belief(user_id, tgt_node.name, likelihood_evidence=conf)

                # Parse timestamp safely
                ts_str = edge_info.get("timestamp")
                valid_from = datetime.now(timezone.utc)
                if ts_str:
                    try:
                        valid_from = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                    except Exception:
                        valid_from = datetime.now(timezone.utc)

                await graph_mem.update_edge(
                    user_id=user_id,
                    source_id=src_node.id,
                    target_id=tgt_node.id,
                    relationship=edge_info.get("relationship", "associated_with"),
                    valid_from=valid_from,
                    session_id=session_id,
                    agent_id=agent_id
                )

            if facts or edges:
                log_graph(f"Background extraction persisted [bold green]{len(facts)}[/bold green] facts and [bold green]{len(edges)}[/bold green] graph edges")

        except Exception as e:
            log_error(f"Background extraction handled error: {e}")
