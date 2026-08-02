"""Pydantic schemas for session logging & chat transcript layer."""

from uuid import UUID
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class MessageSchema(BaseModel):
    """Message item in immutable audit transcript."""
    id: Optional[UUID] = None
    session_id: UUID
    role: str = Field(..., description="Role: 'user', 'assistant', 'system', 'tool'")
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class SessionSchema(BaseModel):
    """Session container item."""
    id: Optional[UUID] = None
    user_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
