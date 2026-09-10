"""Pydantic models for the query/answer pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single citation linking a claim to its source."""
    document_id: UUID
    document_name: str
    chunk_id: UUID
    page_number: Optional[int] = None
    section: Optional[str] = None
    content_snippet: str
    relevance_score: float = 0.0


class QueryRequest(BaseModel):
    """Incoming query from the user."""
    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: Optional[List[UUID]] = None  # filter to specific docs
    conversation_id: Optional[UUID] = None
    top_k: Optional[int] = None  # override default retrieval count
    use_agent: Optional[bool] = None  # Per-request agentic RAG override (True=force agentic, False=standard RAG)
    force_route: Optional[str] = None  # Per-request route override ("rag", "structured_data", "visual", "agentic")


class QueryResponse(BaseModel):
    """Full answer response with citations and metadata."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    query: str
    conversation_id: Optional[UUID] = None
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationMessage(BaseModel):
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    citations: List[Citation] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
