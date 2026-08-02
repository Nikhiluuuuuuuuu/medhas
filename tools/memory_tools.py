"""Tool definitions and function schemas for LLM tool-calling loop."""

from typing import List, Dict, Any
from memory.working import update_block, create_memory_block, delete_memory_block, append_to_memory_block, audit_memory_doctor
from memory.graph import query_point_in_time, query_subgraph
from memory.atomic import search_facts
from utils import measure_latency, log_tool, log_error

MEMORY_TOOLS_DECLARATION: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "update_scratchpad",
            "description": "Updates the working memory scratchpad block with current goals, reasoning notes, or active context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The updated text content for the scratchpad."}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "Updates the persistent user profile block in working memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The updated user profile text."}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_memory_block",
            "description": "Letta Omni-Tool: Creates a new named RAM block in core memory to track specific user context, domain rules, or emotional/metacognitive state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Unique label for the memory block, e.g. 'tech_stack', 'project_rules'"},
                    "description": {"type": "string", "description": "Purpose of this memory block"},
                    "value": {"type": "string", "description": "Initial text content value"}
                },
                "required": ["label", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory_block",
            "description": "Letta Omni-Tool: Deletes an unused or obsolete named memory block from core memory RAM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Label of the memory block to delete"}
                },
                "required": ["label"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "audit_memory_doctor",
            "description": "Letta Memory Doctor: Audits core memory blocks for token bloat, fragmentation, and optimization recommendations.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_search",
            "description": "Searches long-term archival vector memory for past facts, preferences, or technical details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic search query to search long-term memory."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_historical_graph",
            "description": "Queries historical entity relationships valid at a past date (point-in-time temporal query).",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Target entity name to query historical relationships for"},
                    "iso_timestamp": {"type": "string", "description": "ISO timestamp string, e.g. '2024-06-01T00:00:00Z'"}
                },
                "required": ["entity_name", "iso_timestamp"]
            }
        }
    }
]

async def execute_tool_call(user_id: str, tool_name: str, args: Dict[str, Any]) -> str:
    """Execute a function tool call invoked by the LLM."""
    async with measure_latency(f"tools.execute_tool_call ({tool_name})"):
        try:
            if tool_name == "update_scratchpad":
                text = args.get("text", "")
                await update_block(user_id, "scratchpad", text)
                return f"Successfully updated scratchpad block to: '{text}'"

            elif tool_name == "update_user_profile":
                text = args.get("text", "")
                await update_block(user_id, "user_profile", text)
                return f"Successfully updated user profile to: '{text}'"

            elif tool_name == "create_memory_block":
                label = args.get("label", "")
                desc = args.get("description", "")
                val = args.get("value", "")
                res = await create_memory_block(user_id, label, desc, val)
                return f"Successfully created memory block '{label}': {res}"

            elif tool_name == "delete_memory_block":
                label = args.get("label", "")
                res = await delete_memory_block(user_id, label)
                return f"Successfully deleted memory block '{label}'"

            elif tool_name == "audit_memory_doctor":
                res = await audit_memory_doctor(user_id)
                return f"Memory doctor audit results: {res}"

            elif tool_name == "archival_memory_search":
                query = args.get("query", "")
                facts = await search_facts(user_id, query, limit=5)
                if not facts:
                    return f"No relevant archival memory facts found for query: '{query}'"
                fact_lines = [f"- {f.fact_text} (similarity: {f.similarity:.2f})" for f in facts]
                return f"Archival memory search results for '{query}':\n" + "\n".join(fact_lines)

            elif tool_name == "query_historical_graph":
                from datetime import datetime
                entity_name = args.get("entity_name", "")
                iso_ts = args.get("iso_timestamp", "")
                dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
                results = await query_point_in_time(user_id, entity_name, dt)
                return f"Historical graph query for {entity_name} at {iso_ts}: {results}"

            else:
                return f"Unknown tool name: '{tool_name}'"
        except Exception as e:
            log_error(f"Error executing tool {tool_name}: {e}")
            return f"Tool execution failed: {e}"
