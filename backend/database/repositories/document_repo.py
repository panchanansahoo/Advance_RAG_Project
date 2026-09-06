"""CRUD operations for Document records."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import Document
from backend.schemas.common import ProcessingStatus


class DocumentRepository:
    """Async repository wrapping Document table operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, document: Document) -> Document:
        self.session.add(document)
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def get_by_id(self, document_id: UUID) -> Optional[Document]:
        result = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_all(
        self, skip: int = 0, limit: int = 50
    ) -> tuple[List[Document], int]:
        # Count
        count_result = await self.session.execute(
            select(func.count(Document.id))
        )
        total = count_result.scalar_one()

        # Fetch
        result = await self.session.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        documents = list(result.scalars().all())
        return documents, total

    async def update_status(
        self,
        document_id: UUID,
        status: ProcessingStatus,
        chunk_count: int = 0,
        page_count: int = 0,
        error_message: Optional[str] = None,
    ) -> Optional[Document]:
        doc = await self.get_by_id(document_id)
        if doc is None:
            return None
        doc.status = status
        if chunk_count:
            doc.chunk_count = chunk_count
        if page_count:
            doc.page_count = page_count
        if error_message:
            doc.error_message = error_message
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def delete(self, document_id: UUID) -> bool:
        doc = await self.get_by_id(document_id)
        if doc is None:
            return False
        await self.session.delete(doc)
        await self.session.commit()
        return True
