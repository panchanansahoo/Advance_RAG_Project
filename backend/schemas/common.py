"""Shared types, enums, and base models used across the application."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Generic, List, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────

class ContentType(str, Enum):
    """Type of content within a document chunk."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    DIAGRAM = "diagram"
    CODE = "code"
    STRUCTURED = "structured"


class ProcessingStatus(str, Enum):
    """Status of a document through the ingestion pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Supported document file types."""
    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "md"
    CSV = "csv"
    XLSX = "xlsx"
    DOCX = "docx"
    IMAGE = "image"
    HTML = "html"


# ── Pagination ──────────────────────────────────────────────

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""
    items: List[T]
    total: int
    page: int = 1
    page_size: int = 20
    has_more: bool = False


# ── Timestamps ──────────────────────────────────────────────

class TimestampMixin(BaseModel):
    """Mixin adding created_at / updated_at timestamps."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# ── Error Response ──────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response body."""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
