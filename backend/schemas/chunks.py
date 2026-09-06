"""Pydantic models for chunks — the normalized internal representation (PRD §8)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from backend.schemas.common import ContentType


class ChunkCreate(BaseModel):
    """Data required to create and store a chunk."""
    document_id: UUID
    chunk_index: int
    content: str
    content_type: ContentType = ContentType.TEXT
    page_number: Optional[int] = None
    section: Optional[str] = None
    table_data: Optional[Dict[str, Any]] = None
    visual_description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = None


class ChunkResponse(BaseModel):
    """Chunk record returned by the API / used in retrieval results."""
    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    content_type: ContentType
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = {"from_attributes": True}


class RetrievedChunk(BaseModel):
    """A chunk returned by the retrieval pipeline, with relevance score."""
    chunk_id: UUID
    document_id: UUID
    content: str
    content_type: ContentType
    page_number: Optional[int] = None
    section: Optional[str] = None
    score: float = 0.0
    source_filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
