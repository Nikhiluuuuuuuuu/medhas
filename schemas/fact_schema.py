"""Pydantic schemas for Mem0 & GBrain atomic facts, importance scoring, and vector lifecycle."""

from uuid import UUID
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AtomicFactSchema(BaseModel):
    """Atomic fact with vector embedding, importance score, and active lifecycle state.

    Mirrors Mem0's fact record: supports `categories`, `memory_type`
    (semantic/episodic/procedural), `run_id` scoping, and free-form `metadata`.
    """
    id: Optional[UUID] = None
    user_id: str
    session_id: Optional[UUID] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    fact_text: str
    categories: List[str] = Field(default_factory=list)
    memory_type: str = "semantic"   # semantic | episodic | procedural
    metadata: Dict[str, Any] = Field(default_factory=dict)
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
    fts_rank: float = 0.0  # true Postgres BM25/FTS score (ts_rank_cd) — used in fusion rerank
    importance_score: float
    created_at: datetime
    session_id: Optional[UUID] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    memory_type: str = "semantic"
    metadata: Dict[str, Any] = Field(default_factory=dict)
