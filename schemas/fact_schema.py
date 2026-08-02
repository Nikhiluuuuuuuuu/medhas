"""Pydantic schemas for Mem0 & GBrain atomic facts, importance scoring, and vector lifecycle."""

from uuid import UUID
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class AtomicFactSchema(BaseModel):
    """Atomic fact with vector embedding, importance score, and active lifecycle state."""
    id: Optional[UUID] = None
    user_id: str
    session_id: Optional[UUID] = None
    agent_id: Optional[str] = None
    fact_text: str
    embedding: Optional[List[float]] = None
    importance_score: float = Field(default=5.0, description="Importance score from 1.0 (trivial) to 10.0 (critical)")
    is_active: bool = True
    created_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None


class FactSearchResult(BaseModel):
    """Result from RRF hybrid vector search & recency decay scoring."""
    id: UUID
    fact_text: str
    similarity: float
    rrf_score: float
    importance_score: float
    created_at: datetime
    session_id: Optional[UUID] = None
    agent_id: Optional[str] = None
