"""
SQLAlchemy ORM models mapping to PostgreSQL tables.
Covers: documents, chunks, conversations, messages, citations.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import relationship

from backend.database.connection import Base
from backend.schemas.common import ContentType, DocumentType, ProcessingStatus


class Document(Base):
    """Uploaded document metadata and processing status."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    title = Column(String(512), nullable=True)
    description = Column(Text, nullable=True)
    document_type = Column(SAEnum(DocumentType), nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_size = Column(Integer, nullable=False, default=0)
    status = Column(
        SAEnum(ProcessingStatus),
        nullable=False,
        default=ProcessingStatus.PENDING,
    )
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    tags = Column(JSON, default=list)
    metadata_ = Column("metadata", JSON, default=dict)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.utcnow, nullable=True)

    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """A single content chunk extracted from a document."""

    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(
        SAEnum(ContentType), nullable=False, default=ContentType.TEXT
    )
    page_number = Column(Integer, nullable=True)
    section = Column(String(512), nullable=True)
    table_data = Column(JSON, nullable=True)
    visual_description = Column(Text, nullable=True)
    token_count = Column(Integer, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)
    embedding_id = Column(String(256), nullable=True)  # ID in vector store
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    """A chat conversation session."""

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.utcnow, nullable=True)

    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    """A single message within a conversation."""

    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    citations = Column(JSON, default=list)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")


class CitationRecord(Base):
    """Persistent citation linking an answer claim to source evidence."""

    __tablename__ = "citations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id = Column(UUID(as_uuid=True), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), nullable=False)
    page_number = Column(Integer, nullable=True)
    section = Column(String(512), nullable=True)
    content_snippet = Column(Text, nullable=True)
    relevance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
