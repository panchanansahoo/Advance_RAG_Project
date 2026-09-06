"""CRUD operations for Chunk records."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import Chunk


class ChunkRepository:
    """Async repository wrapping Chunk table operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_many(self, chunks: List[Chunk]) -> List[Chunk]:
        self.session.add_all(chunks)
        await self.session.commit()
        for chunk in chunks:
            await self.session.refresh(chunk)
        return chunks

    async def get_by_id(self, chunk_id: UUID) -> Optional[Chunk]:
        result = await self.session.execute(
            select(Chunk).where(Chunk.id == chunk_id)
        )
        return result.scalar_one_or_none()

    async def get_by_document(self, document_id: UUID) -> List[Chunk]:
        result = await self.session.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())

    async def get_by_ids(self, chunk_ids: List[UUID]) -> List[Chunk]:
        if not chunk_ids:
            return []
        result = await self.session.execute(
            select(Chunk).where(Chunk.id.in_(chunk_ids))
        )
        return list(result.scalars().all())

    async def delete_by_document(self, document_id: UUID) -> int:
        result = await self.session.execute(
            sa_delete(Chunk).where(Chunk.document_id == document_id)
        )
        await self.session.commit()
        return result.rowcount
