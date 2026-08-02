"""Pydantic schemas for Letta-style working memory prompt RAM.

Inspired by Letta/MemGPT: working memory is a *registry of labeled blocks*
the agent self-edits, not a fixed set of four fields. This allows arbitrary
custom blocks (e.g. ``tech_stack``) to persist alongside the canonical ones.
"""

from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field


class MemoryBlock(BaseModel):
    """A single labeled working-memory block (Letta/MemGPT core primitive)."""
    label: str = Field(..., description="Unique block label, e.g. 'user_profile'.")
    description: str = Field(default="", description="Purpose of this block.")
    value: str = Field(default="", description="Current text content of the block.")
    limit_tokens: int = Field(default=1000, description="Soft token budget for the block.")


# Canonical block labels every user gets by default.
DEFAULT_BLOCKS: Dict[str, MemoryBlock] = {
    "persona": MemoryBlock(
        label="persona",
        description="Core system persona and instructions.",
        value="I am an advanced AI assistant powered by the Unified Multi-AGI Memory Engine.",
    ),
    "user_profile": MemoryBlock(
        label="user_profile",
        description="Persistent factual profile of the user.",
        value="User profile is empty.",
    ),
    "scratchpad": MemoryBlock(
        label="scratchpad",
        description="Transient workspace for current conversation reasoning.",
        value="No active scratchpad notes.",
    ),
    "active_goals": MemoryBlock(
        label="active_goals",
        description="Current user goals and agent tasks.",
        value="No active goals set.",
    ),
}


class WorkingMemoryBlocks(BaseModel):
    """Registry of labeled memory blocks keyed by lowercase label.

    ``extra="allow"`` lets arbitrary custom blocks persist (BUG-1 fix); the
    schema no longer silently drops unknown block labels on re-validation.
    """
    model_config = {"extra": "allow"}

    persona: str = Field(
        default=DEFAULT_BLOCKS["persona"].value,
        description="Core system persona and instructions.",
    )
    user_profile: str = Field(
        default=DEFAULT_BLOCKS["user_profile"].value,
        description="Persistent factual profile of the user.",
    )
    scratchpad: str = Field(
        default=DEFAULT_BLOCKS["scratchpad"].value,
        description="Transient workspace for current conversation reasoning.",
    )
    active_goals: str = Field(
        default=DEFAULT_BLOCKS["active_goals"].value,
        description="Current user goals and agent tasks.",
    )

    # --- Block registry helpers (the real persistence layer) ---
    def to_block_map(self) -> Dict[str, MemoryBlock]:
        """Return every block (canonical + custom) as a {label: MemoryBlock} map.

        Custom blocks are stored as full ``MemoryBlock`` dicts in the extra
        namespace (``__pydantic_extra__``); canonical four are also mirrored
        there by ``from_block_map``. We read exclusively from the extra
        namespace so no block content is lost on round-trips.
        """
        blocks: Dict[str, MemoryBlock] = {}
        extra = getattr(self, "__pydantic_extra__", None) or {}
        for label, raw in extra.items():
            if isinstance(raw, MemoryBlock):
                blocks[label] = raw
            elif isinstance(raw, dict) and "value" in raw:
                blocks[label] = MemoryBlock(**raw)
            elif isinstance(raw, str):
                blocks[label] = MemoryBlock(label=label, value=raw)
        # Fallback: include canonical top-level string fields if absent.
        for label in ("persona", "user_profile", "scratchpad", "active_goals"):
            if label not in blocks:
                val = getattr(self, label, None)
                if val:
                    blocks[label] = MemoryBlock(label=label, value=val)
        return blocks

    @classmethod
    def from_block_map(cls, blocks: Dict[str, MemoryBlock]) -> "WorkingMemoryBlocks":
        """Build a registry from a {label: MemoryBlock} map.

        Canonical four labels are stored as top-level string fields for
        backwards compatibility; every block (including custom) is also kept
        as a full ``MemoryBlock`` in the extra namespace so no content is lost.
        """
        kwargs: Dict[str, str] = {}
        extra: Dict[str, object] = {}
        for label, block in blocks.items():
            if label in DEFAULT_BLOCKS:
                kwargs[label] = block.value
            extra[label] = block.model_dump()
        obj = cls(**kwargs)
        obj.__pydantic_extra__ = extra
        return obj


class WorkingMemoryRecord(BaseModel):
    """Database record for user working memory."""
    user_id: str
    blocks: WorkingMemoryBlocks
    updated_at: Optional[datetime] = None
