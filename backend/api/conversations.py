"""
Conversations API — CRUD endpoints for managing chat conversations and history.

PRD §20: Conversation Memory.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db
from backend.database.repositories.conversation_repo import ConversationRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/conversations", tags=["Conversations"])


# ── Schemas ─────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"


class ConversationUpdate(BaseModel):
    title: str


class MessageOut(BaseModel):
    id: UUID
    role: str
    content: str
    citations: list = Field(default_factory=list)
    created_at: str


class ConversationOut(BaseModel):
    id: UUID
    title: Optional[str]
    created_at: str
    updated_at: Optional[str] = None
    messages: List[MessageOut] = Field(default_factory=list)


class ConversationListOut(BaseModel):
    conversations: List[ConversationOut]
    total: int


# ── Endpoints ───────────────────────────────────────────────

@router.post("/", response_model=ConversationOut, status_code=201)
async def create_conversation(
    body: ConversationCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation."""
    repo = ConversationRepository(db)
    conv = await repo.create_conversation(title=body.title)
    return ConversationOut(
        id=conv.id,
        title=conv.title,
        created_at=str(conv.created_at),
    )


@router.get("/", response_model=ConversationListOut)
async def list_conversations(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List all conversations, newest first."""
    repo = ConversationRepository(db)
    conversations, total = await repo.list_conversations(skip=skip, limit=limit)
    return ConversationListOut(
        conversations=[
            ConversationOut(
                id=c.id,
                title=c.title,
                created_at=str(c.created_at),
                updated_at=str(c.updated_at) if c.updated_at else None,
            )
            for c in conversations
        ],
        total=total,
    )


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a conversation with all its messages."""
    repo = ConversationRepository(db)
    conv = await repo.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationOut(
        id=conv.id,
        title=conv.title,
        created_at=str(conv.created_at),
        updated_at=str(conv.updated_at) if conv.updated_at else None,
        messages=[
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                citations=m.citations or [],
                created_at=str(m.created_at),
            )
            for m in sorted(conv.messages, key=lambda m: m.created_at)
        ],
    )


@router.put("/{conversation_id}", response_model=ConversationOut)
async def update_conversation(
    conversation_id: UUID,
    body: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a conversation's title."""
    repo = ConversationRepository(db)
    conv = await repo.update_conversation_title(conversation_id, body.title)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationOut(
        id=conv.id,
        title=conv.title,
        created_at=str(conv.created_at),
        updated_at=str(conv.updated_at) if conv.updated_at else None,
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation and all its messages."""
    repo = ConversationRepository(db)
    deleted = await repo.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
