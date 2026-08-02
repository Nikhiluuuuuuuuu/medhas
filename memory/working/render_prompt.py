"""Layer 2 (Letta): Render formatted system prompt from working memory blocks."""

from schemas import WorkingMemoryBlocks
from memory.working.get_blocks import get_blocks
from utils import measure_latency

async def render_prompt(user_id: str) -> str:
    """Format working memory blocks into a clean structured system prompt."""
    async with measure_latency("memory.working.render_prompt"):
        record = await get_blocks(user_id)
        b = record.blocks

        prompt = f"""
# SYSTEM PERSONA & INSTRUCTIONS
{b.persona}

# MEMORY CONTEXT BLOCKS (DYNAMIC RAM)

<user_profile>
{b.user_profile}
</user_profile>

<active_goals>
{b.active_goals}
</active_goals>

<scratchpad>
{b.scratchpad}
</scratchpad>
""".strip()
        return prompt
