"""Sub-10ms Hot Path Context Retrieval & Dynamic Prompt Assembler."""

import asyncio
import re
from typing import Tuple, List, Dict, Any
from uuid import UUID
from memory.working import render_prompt
from memory.session import get_transcript
from memory.atomic import search_facts
from memory.graph import run_spreading_activation
from config import settings
from utils import measure_latency, log_error

def extract_seed_terms(text: str) -> List[str]:
    """Dynamically extract entity and topic seed terms without any hardcoded stopword lists, using linguistic token analysis."""
    # Extract alphanumeric tokens, technical terms, acronyms, and proper nouns
    raw_tokens = re.findall(r"\b[A-Za-z0-9_\-]{3,}\b", text)
    seeds = []
    seen = set()

    for tok in raw_tokens:
        tok_clean = tok.strip("-_")
        if not tok_clean:
            continue

        # Dynamic Entity Identification: Acronyms, numbers/version symbols, technical compounds, or proper noun casing
        is_acronym_or_tech = tok_clean.isupper() or any(c.isdigit() for c in tok_clean) or "_" in tok_clean or "-" in tok_clean
        is_proper_noun = tok_clean[0].isupper() and not tok_clean.isupper()

        # Retain if technical entity, proper noun, or meaningful length topic candidate
        if is_acronym_or_tech or is_proper_noun or len(tok_clean) >= 4:
            term = tok_clean if tok_clean.isupper() else tok_clean.capitalize()
            if term.lower() not in seen:
                seen.add(term.lower())
                seeds.append(term)

        if len(seeds) >= 5:
            break

    if "user" not in seen:
        seeds.append("User")
    return seeds

async def assemble_context_and_prompt(
    user_id: str,
    session_id: UUID,
    user_message: str
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute retrieval across Working RAM, Atomic Facts, SAN Graph, and Session Transcript.

    Note: fact retrieval is boosted by the graph activation, so spreading activation is
    computed first; system_prompt and transcript are still gathered in parallel with facts.
    """
    async with measure_latency("pipeline.hot_path.assemble_context_and_prompt"):
        # 1. Seed terms + Spreading Activation (HippoRAG PPR over the entity graph)
        seed_terms = extract_seed_terms(user_message)
        activated_graph = await run_spreading_activation(user_id, seed_terms)
        activated_entities = []
        for g in activated_graph:
            for key in ("source_name", "target_name"):
                if g.get(key) and g[key] not in activated_entities:
                    activated_entities.append(g[key])

        # 2. Parallel retrieval across memory layers (facts boosted by activated entities)
        system_prompt_task = render_prompt(user_id)
        facts_task = search_facts(
            user_id, user_message, limit=settings.TOP_K_FACTS,
            apply_rrf=True, boost_entities=activated_entities or None
        )
        transcript_task = get_transcript(session_id, limit=10)

        system_prompt, facts, transcript = await asyncio.gather(
            system_prompt_task,
            facts_task,
            transcript_task
        )

        # Format retrieved context sections
        facts_section = ""
        if facts:
            fact_lines = [f"- {f.fact_text} (RRF Score: {f.rrf_score:.4f})" for f in facts]
            facts_section = "\n\n<retrieved_atomic_facts>\n" + "\n".join(fact_lines) + "\n</retrieved_atomic_facts>"

        graph_section = ""
        if activated_graph:
            graph_lines = [f"- {g['source_name']} --[{g['relationship']}]--> {g['target_name']} (Energy: {g['activation_energy']:.2f})" for g in activated_graph[:5]]
            graph_section = "\n\n<activated_knowledge_graph>\n" + "\n".join(graph_lines) + "\n</activated_knowledge_graph>"

        augmented_system_prompt = system_prompt + facts_section + graph_section

        # Format chat transcript for LLM payload with mandatory tool_call_id for tool messages
        formatted_messages = [{"role": "system", "content": augmented_system_prompt}]
        for msg in transcript:
            item = {"role": msg.role, "content": msg.content}
            if msg.role == "tool":
                tc_id = msg.metadata.get("tool_call_id", "call_default_id") if isinstance(msg.metadata, dict) else "call_default_id"
                item["tool_call_id"] = tc_id
            formatted_messages.append(item)

        # Append current user turn if not already present in transcript
        if not transcript or transcript[-1].role != "user" or transcript[-1].content != user_message:
            formatted_messages.append({"role": "user", "content": user_message})

        return augmented_system_prompt, formatted_messages

