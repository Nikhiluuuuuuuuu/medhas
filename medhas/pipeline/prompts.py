"""Production extraction & retrieval prompts (de-hardcoded from medhas.pipeline code).

Prompts are loaded here so they can be overridden via config/env without touching logic.
Mirrors how Mem0/Cognee keep prompts in dedicated modules rather than inline strings.
"""

EXTRACTION_PROMPT = """\
You are a memory extraction worker for a unified multi-AGI memory engine.
Analyze the user's latest message and the assistant's response, together with the recent \
conversation context, and extract durable memory:

1. New atomic facts or preference updates (one concise sentence each).
2. Entity relationship edges:
   source, source_type, target, target_type, relationship, timestamp (ISO-8601 or null), \
confidence (0.50-0.99).

Return ONLY a JSON object (no prose, no markdown fences):
{
  "facts": ["fact 1", "fact 2"],
  "edges": [
    {"source": "User", "source_type": "Person", "target": "TargetEntity",
     "target_type": "Entity", "relationship": "relationship_type",
     "timestamp": "2024-01-01T00:00:00Z", "confidence": 0.85}
  ]
}
If nothing new, return {"facts": [], "edges": []}.
"""

# Mem0-style last-K context framing appended to the extraction prompt.
CONTEXT_TEMPLATE = """\
=== Recent conversation context (last {k} messages) ===
{context}
=====================================================
"""

# System prompt section appended when rendering the agent context.
MEMORY_SECTION_HEADER = "Retrieved long-term memory"
