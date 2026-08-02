"""Pydantic schemas for Zep/Graphiti-style bi-temporal knowledge graph."""

from uuid import UUID
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class GraphNodeSchema(BaseModel):
    """Entity node in knowledge graph."""
    id: Optional[UUID] = None
    user_id: str
    name: str
    entity_type: str = Field(..., description="Entity type, e.g., 'Person', 'Company', 'Location'")
    attributes: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class GraphEdgeSchema(BaseModel):
    """Bi-temporal relationship edge in knowledge graph."""
    id: Optional[UUID] = None
    user_id: str
    source_id: UUID
    target_id: UUID
    relationship: str
    valid_from: datetime
    valid_to: Optional[datetime] = None  # None indicates active edge
    created_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None


class SubgraphQueryResult(BaseModel):
    """Entity node with its valid connected relationships."""
    node: GraphNodeSchema
    outgoing_edges: List[Dict[str, Any]] = Field(default_factory=list)
    incoming_edges: List[Dict[str, Any]] = Field(default_factory=list)
