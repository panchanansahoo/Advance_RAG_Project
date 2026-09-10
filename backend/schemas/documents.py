"""Pydantic models for document upload, listing, and status."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from backend.schemas.common import DocumentType, ProcessingStatus


class DocumentUpload(BaseModel):
    """Metadata sent alongside a file upload."""
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    """Document record returned by the API."""
    id: UUID
    filename: str
    title: Optional[str] = None
    description: Optional[str] = None
    document_type: DocumentType
    file_size: int
    status: ProcessingStatus
    error_message: Optional[str] = None
    chunk_count: int = 0
    page_count: int = 0
    tags: List[str] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    documents: List[DocumentResponse]
    total: int


class DocumentStatusResponse(BaseModel):
    """Lightweight status check response."""
    id: UUID
    status: ProcessingStatus
    chunk_count: int = 0
    error_message: Optional[str] = None
