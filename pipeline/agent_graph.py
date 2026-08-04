"""Master Unified AGI Memory Engine Orchestrator with Metacognitive Dual-Process Control."""

import asyncio
import json
from uuid import UUID, uuid4
from typing import Dict, Any, Optional
import memory.session as session_mem
from infrastructure.llm import GroqLLMProvider
from tools import MEMORY_TOOLS_DECLARATION, execute_tool_call
from pipeline.hot_path import assemble_context_and_prompt
from pipeline.async_extractor import extract_and_persist_background
from pipeline.metacognition import evaluate_cognitive_mode
from utils import measure_latency, log_tool, log_error
from config import settings

# AGI roadmap engine (E1–E37). Wired into the live loop when AGI_ENGINE_ENABLED.
# Kept additive: the original 6-in-1 hot path still runs; the AGI engine augments
# recall (metamemory/abstention/routing) and writes new memories through the full pipeline.
from agi.engine import engine as agi_engine

class UnifiedMemoryEngine:
    """Master production engine running 6-in-1 unified local agent memory system."""

    def __init__(self, llm_provider: Optional[GroqLLMProvider] = None):
        self.llm = llm_provider or GroqLLMProvider()

    async def execute_turn(
        self,
        user_id: str,
        session_id: UUID,
        user_message: str
    ) -> str:
        """Execute a full conversation turn with metacognitive System 1 vs System 2 control."""
        async with measure_latency("UnifiedMemoryEngine.execute_turn"):
            # 1. Ensure Session exists in DB (creates on-the-fly if missing) and Log User Turn (Layer 1: Convex)
            await session_mem.ensure_session_exists(session_id, user_id)
            await session_mem.log_message(session_id, "user", user_message)


            # 2. Metacognitive Strategy Evaluation
            mode, meta_context = await evaluate_cognitive_mode(user_id, user_message)

            # Fast Path System 1 execution if procedural playbook is available
            if mode == "SYSTEM_1" and "playbook" in meta_context:
                pb = meta_context["playbook"]
                steps_str = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(pb["steps"]))
                final_text = f"Executed procedural skill playbook for task '{pb['task']}':\n{steps_str}"
                await session_mem.log_message(session_id, "assistant", final_text)
                return final_text

            # 3. HOT PATH: Parallel Context Retrieval & System Prompt Assembly (<10ms)
            if getattr(settings, "AGI_ENGINE_ENABLED", True):
                system_prompt, messages = await self.agi_context(user_id, session_id, user_message)
            else:
                system_prompt, messages = await assemble_context_and_prompt(user_id, session_id, user_message)

            # 4. LLM Completion & Tool Call Execution Loop
            llm_response = await self.llm.chat_completion(messages, tools=MEMORY_TOOLS_DECLARATION)
            tool_calls = llm_response.get("tool_calls", [])

            # Handle tool calls mid-session if requested by LLM
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": llm_response.get("content") or "",
                    "tool_calls": tool_calls
                })

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                    
                    tool_output = await execute_tool_call(user_id, fn_name, fn_args)
                    log_tool(fn_name, tool_output)
                    
                    # Log tool message turn with tool_call_id
                    tc_id = tc.get("id", f"call_{fn_name}")
                    await session_mem.log_message(
                        session_id,
                        "tool",
                        f"Tool [{fn_name}] executed: {tool_output}",
                        metadata={"tool_name": fn_name, "tool_call_id": tc_id}
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": str(tool_output)
                    })

                # Re-query LLM for final natural response after tool execution
                second_response = await self.llm.chat_completion(messages, temperature=0.3)
                final_text = second_response.get("content", "").strip() or "Understood. Updated working memory context."
            else:
                final_text = llm_response.get("content", "").strip() or "Understood. Context analyzed."

            # 5. Log Assistant Turn (Layer 1: Convex)
            await session_mem.log_message(session_id, "assistant", final_text)

            # 6. ASYNC PATH: Fire non-blocking Cognee ECL background extraction task.
            # Supervised fire-and-forget: exceptions are logged rather than silently lost.
            task = asyncio.create_task(
                extract_and_persist_background(user_id, user_message, final_text, session_id=session_id)
            )
            task.add_done_callback(_log_task_failure)

            return final_text

    async def agi_context(
        self,
        user_id: str,
        session_id: UUID,
        user_message: str,
    ) -> "tuple[str, list]":
        """AGI-augmented context assembly (E1–E37).

        Uses the unified engine for recall so the prompt carries calibrated confidence
        and abstention signals, while still rendering the working-memory system prompt
        and transcript the same way the 6-in-1 hot path does.
        """
        async with measure_latency("UnifiedMemoryEngine.agi_context"):
            from memory.working import render_prompt
            from memory.session import get_transcript

            system_prompt_task = render_prompt(user_id)
            recall_task = agi_engine.recall(user_id, user_message, enforce_abstention=False)
            transcript_task = get_transcript(session_id, limit=10)
            system_prompt, recall, transcript = await asyncio.gather(
                system_prompt_task, recall_task, transcript_task
            )

            facts_section = ""
            if recall.get("status") == "abstained":
                facts_section = (
                    "\n\n<retrieved_atomic_facts confidence=\"low\">\n"
                    "[No sufficiently confident memory found — do not answer from memory; "
                    "ask or state you don't know.]\n</retrieved_atomic_facts>"
                )
            elif recall.get("results"):
                lines = [f"- {r.get('fact_text', '')}" for r in recall["results"]]
                facts_section = "\n\n<retrieved_atomic_facts>\n" + "\n".join(lines) + "\n</retrieved_atomic_facts>"

            augmented_system_prompt = system_prompt + facts_section

            formatted_messages = [{"role": "system", "content": augmented_system_prompt}]
            for msg in transcript:
                item = {"role": msg.role, "content": msg.content}
                if msg.role == "tool":
                    tc_id = msg.metadata.get("tool_call_id", "call_default_id") if isinstance(msg.metadata, dict) else "call_default_id"
                    item["tool_call_id"] = tc_id
                formatted_messages.append(item)

            if not transcript or transcript[-1].role != "user" or transcript[-1].content != user_message:
                formatted_messages.append({"role": "user", "content": user_message})

            return augmented_system_prompt, formatted_messages

    async def capture(
        self,
        user_id: str,
        content: str,
        *,
        session_id: Optional[UUID] = None,
        agent_id: Optional[str] = None,
        background: bool = True,
    ) -> Dict[str, Any]:
        """GBrain `capture` equivalent: single entrypoint to get content into the brain.

        Routes the content through the same ingest path as a conversation turn:
        logs it as a session message (or a standalone message if no session), runs the
        hot-path retrieval, and schedules the Cognee-style background extraction that
        populates atomic facts + the bi-temporal graph. Returns a summary handle so the
        caller can correlate.

        `background=False` runs extraction synchronously (useful for tests/bulk ingest).
        """
        async with measure_latency("UnifiedMemoryEngine.capture"):
            sid = session_id or uuid4()
            await session_mem.ensure_session_exists(sid, user_id)
            await session_mem.log_message(sid, "user", content)

            if not background:
                await extract_and_persist_background(user_id, content, "", session_id=sid, agent_id=agent_id)
                return {"status": "ingested", "session_id": str(sid)}

            task = asyncio.create_task(
                extract_and_persist_background(user_id, content, "", session_id=sid, agent_id=agent_id)
            )
            task.add_done_callback(_log_task_failure)
            return {"status": "scheduled", "session_id": str(sid)}


def _log_task_failure(task: "asyncio.Task[None]") -> None:
    """Surface exceptions from supervised background tasks (no silent failures)."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log_error(f"Background extraction task failed: {exc}")
