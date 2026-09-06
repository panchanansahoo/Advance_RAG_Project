"""
Ingestion service — orchestrates the full document processing pipeline:
File → Validate → Save → Process → Chunk → Embed → Store in Vector DB + PostgreSQL.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import get_settings
from backend.database.models import Chunk as ChunkModel, Document
from backend.embeddings import get_embedding_provider
from backend.ingestion.file_validator import FileValidator
from backend.processing.chunking import get_chunker
from backend.processing.processor_factory import get_processor
from backend.retrieval.vector_store import get_vector_store
from backend.schemas.common import ProcessingStatus

logger = logging.getLogger(__name__)


class IngestionService:
    """
    End-to-end document ingestion pipeline.
    Called as a background task after the upload endpoint accepts the file.
    """

    def __init__(self, db_session):
        self.db_session = db_session
        self.settings = get_settings()
        self.validator = FileValidator()

    async def process_document(self, document_id: uuid.UUID) -> None:
        """
        Process a previously uploaded document through the full pipeline.

        This runs as a background task:
        1. Load document record from DB
        2. Extract content using the appropriate processor
        3. Chunk the extracted content
        4. Generate embeddings
        5. Store in vector DB with payloads
        6. Save chunk records to PostgreSQL
        7. Update document status
        """
        from backend.database.repositories import DocumentRepository, ChunkRepository

        doc_repo = DocumentRepository(self.db_session)
        chunk_repo = ChunkRepository(self.db_session)

        # Load the document
        document = await doc_repo.get_by_id(document_id)
        if not document:
            logger.error("Document %s not found", document_id)
            return

        try:
            # Update status to processing
            await doc_repo.update_status(document_id, ProcessingStatus.PROCESSING)

            # 1. Get the processor for this file type
            ext = Path(document.file_path).suffix.lower()
            processor = get_processor(ext)
            if not processor:
                raise ValueError(f"No processor available for extension: {ext}")

            # 2. Extract content
            logger.info("Processing document: %s (%s)", document.filename, ext)
            processed_contents = await processor.process(
                file_path=document.file_path,
                metadata={
                    "document_id": str(document_id),
                    "filename": document.filename,
                },
            )

            if not processed_contents:
                raise ValueError("No content extracted from document")

            # 3. Chunk the content
            chunker = get_chunker(
                strategy=self.settings.chunk_strategy,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )

            all_chunk_data = []
            chunk_index = 0
            page_count = 0

            for content in processed_contents:
                chunks = chunker.chunk(
                    text=content.text,
                    page_number=content.page_number,
                    section=content.section,
                    content_type=content.content_type,
                    start_index=chunk_index,
                )
                all_chunk_data.extend(chunks)
                chunk_index += len(chunks)
                if content.page_number and content.page_number > page_count:
                    page_count = content.page_number

            if not all_chunk_data:
                raise ValueError("No chunks generated from content")

            logger.info("Generated %d chunks from document", len(all_chunk_data))

            # 4. Generate embeddings
            embedding_provider = get_embedding_provider()
            texts = [chunk.content for chunk in all_chunk_data]

            # Batch embeddings (process in batches of 64)
            all_embeddings = []
            batch_size = 64
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = await embedding_provider.embed(batch)
                all_embeddings.extend(batch_embeddings)

            # 5. Store in vector DB
            vector_store = get_vector_store()
            await vector_store.initialize(
                self.settings.qdrant_collection,
                embedding_provider.dimension,
            )

            vector_ids = []
            vector_payloads = []
            chunk_models = []

            for i, chunk_data in enumerate(all_chunk_data):
                chunk_id = uuid.uuid4()
                vector_id = str(chunk_id)
                vector_ids.append(vector_id)

                # Payload stored alongside the vector for retrieval
                vector_payloads.append({
                    "chunk_id": str(chunk_id),
                    "document_id": str(document_id),
                    "content": chunk_data.content,
                    "content_type": chunk_data.content_type.value if hasattr(chunk_data.content_type, 'value') else str(chunk_data.content_type),
                    "page_number": chunk_data.page_number,
                    "section": chunk_data.section,
                    "source_filename": document.filename,
                    "chunk_index": chunk_data.chunk_index,
                })

                # DB model
                chunk_models.append(
                    ChunkModel(
                        id=chunk_id,
                        document_id=document_id,
                        chunk_index=chunk_data.chunk_index,
                        content=chunk_data.content,
                        content_type=chunk_data.content_type,
                        page_number=chunk_data.page_number,
                        section=chunk_data.section,
                        token_count=chunk_data.token_count,
                        embedding_id=vector_id,
                        metadata_={
                            "source_filename": document.filename,
                        },
                    )
                )

            await vector_store.add(
                ids=vector_ids,
                embeddings=all_embeddings,
                payloads=vector_payloads,
            )

            # 6. Save chunks to PostgreSQL
            await chunk_repo.create_many(chunk_models)

            # 7. Update document status
            await doc_repo.update_status(
                document_id,
                ProcessingStatus.COMPLETED,
                chunk_count=len(chunk_models),
                page_count=page_count,
            )

            # 8. Update BM25 index with new chunks (Phase 2)
            try:
                from backend.retrieval.bm25_search import get_bm25_index
                bm25_index = get_bm25_index()
                bm25_index.add_chunks(vector_payloads)
                logger.info("BM25 index updated with %d new chunks", len(vector_payloads))
            except Exception as bm25_err:
                logger.warning("BM25 index update failed (non-critical): %s", bm25_err)

            # 9. Extract Knowledge Graph (Phase 5) — process in batches (Task 2.2)
            if self.settings.graph_extraction_enabled:
                try:
                    from backend.processing.graph_extractor import GraphExtractor
                    graph_extractor = GraphExtractor()
                    logger.info("Starting Graph Extraction for document %s", document_id)

                    # Process chunks in batches to cover the full document
                    # while keeping per-call LLM token costs manageable
                    graph_batch_size = getattr(self.settings, 'graph_batch_size', 10)
                    max_graph_chunks = getattr(self.settings, 'graph_max_chunks', 50)
                    chunks_to_process = all_chunk_data[:max_graph_chunks]

                    for batch_start in range(0, len(chunks_to_process), graph_batch_size):
                        batch_end = batch_start + graph_batch_size
                        batch = chunks_to_process[batch_start:batch_end]
                        batch_text = "\n".join([c.content for c in batch])

                        logger.info(
                            "Graph extraction batch %d-%d / %d",
                            batch_start,
                            min(batch_end, len(chunks_to_process)),
                            len(chunks_to_process),
                        )
                        await graph_extractor.extract_and_store(batch_text, str(document_id))

                except Exception as graph_err:
                    logger.warning("Graph extraction failed (non-critical): %s", graph_err)

            logger.info(
                "Document %s processed successfully: %d chunks, %d pages",
                document.filename,
                len(chunk_models),
                page_count,
            )

        except Exception as e:
            logger.error(
                "Failed to process document %s: %s",
                document_id,
                str(e),
                exc_info=True,
            )
            await doc_repo.update_status(
                document_id,
                ProcessingStatus.FAILED,
                error_message=str(e),
            )
