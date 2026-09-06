"""
FastAPI router for document upload, listing, and management.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database.connection import get_db
from backend.database.models import Document
from backend.database.repositories import DocumentRepository
from backend.ingestion.file_validator import FileValidator, FileValidationError
from backend.ingestion.service import IngestionService
from backend.schemas.common import ProcessingStatus
from backend.schemas.documents import (
    DocumentListResponse,
    DocumentResponse,
    DocumentStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document for processing.

    The document is saved and processing starts in the background.
    Poll GET /api/v1/documents/{id} to check status.
    """
    validator = FileValidator()
    settings = get_settings()

    # Validate file
    try:
        file_size = 0
        # Read the file to get size
        contents = await file.read()
        file_size = len(contents)
        await file.seek(0)

        ext = validator.validate(filename=file.filename, file_size=file_size)
    except FileValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save file to disk
    doc_id = uuid.uuid4()
    upload_dir = settings.upload_path / str(doc_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Create document record
    doc_type = validator.get_document_type(ext)
    document = Document(
        id=doc_id,
        filename=file.filename,
        title=title or Path(file.filename).stem,
        description=description,
        document_type=doc_type,
        file_path=str(file_path),
        file_size=file_size,
        status=ProcessingStatus.PENDING,
        tags=[],
        metadata_={},
    )

    doc_repo = DocumentRepository(db)
    document = await doc_repo.create(document)

    # Start background processing
    background_tasks.add_task(_process_document, doc_id)

    logger.info("Document uploaded: %s (id=%s)", file.filename, doc_id)
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        title=document.title,
        description=document.description,
        document_type=document.document_type,
        file_size=document.file_size,
        status=document.status,
        chunk_count=0,
        page_count=0,
        tags=document.tags or [],
        metadata=document.metadata_ or {},
        created_at=document.created_at,
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List all uploaded documents."""
    doc_repo = DocumentRepository(db)
    documents, total = await doc_repo.list_all(skip=skip, limit=limit)

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                title=doc.title,
                description=doc.description,
                document_type=doc.document_type,
                file_size=doc.file_size,
                status=doc.status,
                chunk_count=doc.chunk_count or 0,
                page_count=doc.page_count or 0,
                tags=doc.tags or [],
                metadata=doc.metadata_ or {},
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
            for doc in documents
        ],
        total=total,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific document by ID."""
    doc_repo = DocumentRepository(db)
    doc = await doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        title=doc.title,
        description=doc.description,
        document_type=doc.document_type,
        file_size=doc.file_size,
        status=doc.status,
        chunk_count=doc.chunk_count or 0,
        page_count=doc.page_count or 0,
        tags=doc.tags or [],
        metadata=doc.metadata_ or {},
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and all its associated data."""
    doc_repo = DocumentRepository(db)
    doc = await doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete vectors from vector store
    try:
        vector_store = get_vector_store_instance()
        await vector_store.delete_by_filter({"document_id": str(document_id)})
    except Exception as e:
        logger.warning("Failed to delete vectors: %s", e)

    # Delete file from disk
    try:
        upload_dir = Path(doc.file_path).parent
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
    except Exception as e:
        logger.warning("Failed to delete file: %s", e)

    # Delete from database (cascades to chunks)
    await doc_repo.delete(document_id)
    logger.info("Document deleted: %s", document_id)


def get_vector_store_instance():
    """Helper to get vector store (avoids circular import)."""
    from backend.retrieval.vector_store import get_vector_store
    return get_vector_store()


async def _process_document(document_id: uuid.UUID):
    """Background task to process a document."""
    from backend.database.connection import _get_session_factory

    session = _get_session_factory()()
    try:
        service = IngestionService(db_session=session)
        await service.process_document(document_id)
    except Exception as e:
        logger.error("Background processing failed for %s: %s", document_id, e)
    finally:
        await session.close()
