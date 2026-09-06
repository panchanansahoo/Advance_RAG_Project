"""CRUD operations for Conversation and Message records."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database.models import Conversation, Message


class ConversationRepository:
    """Async repository wrapping Conversation and Message table operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Conversations ───────────────────────────────────────

    async def create_conversation(self, title: Optional[str] = None) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(title=title or "New Conversation")
        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)
        return conversation

    async def get_conversation(self, conversation_id: UUID) -> Optional[Conversation]:
        """Get a conversation by ID with its messages."""
        result = await self.session.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages))
        )
        return result.scalar_one_or_none()

    async def list_conversations(
        self, skip: int = 0, limit: int = 50
    ) -> tuple[List[Conversation], int]:
        """List all conversations ordered by most recent."""
        count_result = await self.session.execute(
            select(func.count(Conversation.id))
        )
        total = count_result.scalar_one()

        result = await self.session.execute(
            select(Conversation)
            .order_by(Conversation.updated_at.desc().nulls_last(), Conversation.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        conversations = list(result.scalars().all())
        return conversations, total

    async def update_conversation_title(
        self, conversation_id: UUID, title: str
    ) -> Optional[Conversation]:
        """Update a conversation's title."""
        conv = await self.get_conversation(conversation_id)
        if conv is None:
            return None
        conv.title = title
        await self.session.commit()
        await self.session.refresh(conv)
        return conv

    async def delete_conversation(self, conversation_id: UUID) -> bool:
        """Delete a conversation and all its messages (cascade)."""
        result = await self.session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = result.scalar_one_or_none()
        if conv is None:
            return False
        await self.session.delete(conv)
        await self.session.commit()
        return True

    # ── Messages ────────────────────────────────────────────

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        citations: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> Message:
        """Add a message to a conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations or [],
            metadata_=metadata or {},
        )
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        return message

    async def get_recent_messages(
        self, conversation_id: UUID, limit: int = 10
    ) -> List[Message]:
        """
        Get the most recent messages for a conversation.

        Args:
            conversation_id: The conversation to fetch from.
            limit: Maximum number of messages to return (default: 10 = ~5 turns).

        Returns:
            Messages ordered chronologically (oldest first).
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = list(result.scalars().all())
        # Return in chronological order
        messages.reverse()
        return messages
