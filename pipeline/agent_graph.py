"""Master Unified AGI Memory Engine Orchestrator with Metacognitive Dual-Process Control."""

import asyncio
import json
from uuid import UUID
from typing import Dict, Any, Optional
import memory.session as session_mem
from infrastructure.llm import GroqLLMProvider
from tools import MEMORY_TOOLS_DECLARATION, execute_tool_call
from pipeline.hot_path import assemble_context_and_prompt
from pipeline.async_extractor import extract_and_persist_background
from pipeline.metacognition import evaluate_cognitive_mode
from utils import measure_latency, log_tool, log_error

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

            # 6. ASYNC PATH: Fire non-blocking Cognee ECL background extraction task
            asyncio.create_task(extract_and_persist_background(user_id, user_message, final_text, session_id=session_id))

            return final_text
