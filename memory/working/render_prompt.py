"""Layer 2 (Letta): Render formatted system prompt from working memory blocks."""

from schemas import WorkingMemoryBlocks
from memory.working.get_blocks import get_blocks
from utils import measure_latency

async def render_prompt(user_id: str) -> str:
    """Format working memory blocks into a clean structured system prompt.

    Renders the four canonical blocks plus any custom blocks (Letta model),
    so the agent always sees its full working-memory state.
    """
    async with measure_latency("memory.working.render_prompt"):
        record = await get_blocks(user_id)
        blocks = record.blocks.to_block_map()

        b = blocks.get("persona")
        persona = b.value if b else "I am an advanced AI assistant."

        sections = []
        for label in ("user_profile", "active_goals", "scratchpad"):
            blk = blocks.get(label)
            if blk is not None:
                sections.append(f"<{label}>\n{blk.value}\n</{label}>")

        # Include any custom blocks (e.g. tech_stack, project_rules) too.
        for label, blk in blocks.items():
            if label in ("persona", "user_profile", "active_goals", "scratchpad"):
                continue
            sections.append(f"<{label}>\n{blk.value}\n</{label}>")

        prompt = f"""
# SYSTEM PERSONA & INSTRUCTIONS
{persona}

# MEMORY CONTEXT BLOCKS (DYNAMIC RAM)
""".strip()
        if sections:
            prompt = prompt + "\n\n" + "\n\n".join(sections)
        return prompt
