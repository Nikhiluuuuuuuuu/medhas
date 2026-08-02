"""Pydantic schemas for Letta-style working memory prompt RAM."""

from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field


class WorkingMemoryBlocks(BaseModel):
    """Structured prompt RAM blocks editable by the agent."""
    persona: str = Field(
        default="I am an advanced AI assistant powered by a 4-in-1 Unified Local Memory Engine.",
        description="Core system persona and instructions."
    )
    user_profile: str = Field(
        default="User profile is empty.",
        description="Persistent factual profile of the user."
    )
    scratchpad: str = Field(
        default="No active scratchpad notes.",
        description="Transient workspace for current conversation reasoning."
    )
    active_goals: str = Field(
        default="No active goals set.",
        description="Current user goals and agent tasks."
    )


class WorkingMemoryRecord(BaseModel):
    """Database record for user working memory."""
    user_id: str
    blocks: WorkingMemoryBlocks
    updated_at: Optional[datetime] = None
