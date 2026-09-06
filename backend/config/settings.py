"""
Centralized application settings loaded from environment variables / .env file.
Uses Pydantic Settings for validation, type coercion, and defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Master configuration — every setting is overridable via env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────
    app_name: str = "Advanced RAG"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    # ── PostgreSQL ──────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password_change_me"
    postgres_db: str = "advanced_rag"

    @property
    def database_url(self) -> str:
        return "sqlite+aiosqlite:///./advanced_rag.db"

    @property
    def database_url_sync(self) -> str:
        return "sqlite:///./advanced_rag.db"

    # ── Vector Database ─────────────────────────────────────
    vector_db_provider: str = "chroma"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"
    chroma_persist_dir: str = "./chroma_data"

    # ── Embeddings ──────────────────────────────────────────
    embedding_provider: str = "sentence_transformer"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── LLM ─────────────────────────────────────────────────
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ── API Keys ────────────────────────────────────────────
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # ── Ingestion ───────────────────────────────────────────
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    allowed_extensions: List[str] = [
        ".pdf", ".txt", ".md", ".csv", ".xlsx", ".docx",
        ".png", ".jpg", ".jpeg",
    ]

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Chunking ────────────────────────────────────────────
    chunk_strategy: str = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ── Retrieval ───────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_strategy: str = "hybrid"  # "vector", "bm25", "hybrid"

    # ── Hybrid Search (Phase 2) ────────────────────────────
    bm25_enabled: bool = True
    bm25_top_k: int = 20  # BM25 fetches more, then fused
    bm25_weight: float = 0.3  # Weight in RRF fusion
    vector_weight: float = 0.7  # Weight in RRF fusion
    rrf_k: int = 60  # RRF constant (standard default)

    # ── Reranker (Phase 2) ─────────────────────────────────
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5  # Final top-K after reranking

    # ── Query Processing (Phase 2) ─────────────────────────
    query_rewriting_enabled: bool = True
    query_expansion_enabled: bool = False  # Off by default, uses extra LLM calls
    query_decomposition_enabled: bool = False  # For Phase 6 agentic

    # ── Context Compression (Phase 2) ──────────────────────
    context_compression_enabled: bool = True
    max_context_tokens: int = 3000  # Max tokens sent to LLM

    # ── Multimodal Processing (Phase 3) ────────────────────
    ocr_enabled: bool = True
    extract_images_from_pdf: bool = True
    vlm_provider: str = "gemini"  # "gemini" or "openai"
    vlm_model: str = "gemini-1.5-flash"  # e.g., gpt-4o, gemini-1.5-pro

    # ── Agentic & Routing (Phase 4) ────────────────────────
    query_routing_enabled: bool = True
    pandas_execution_timeout: int = 15  # seconds before killing pandas code

    # ── Knowledge Graph (Phase 5) ──────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    graph_extraction_enabled: bool = False  # False by default due to high token cost
    graph_batch_size: int = 10  # Chunks per LLM extraction call
    graph_max_chunks: int = 50  # Max chunks to process per document (cost control)

    # ── Agentic Framework (Phase 6) ────────────────────────
    agentic_rag_enabled: bool = True  # Whether agentic RAG is enabled for adaptive routing
    agentic_rag_force: bool = False  # Global override: if True, forces all queries to agentic
    agent_max_iterations: int = 5  # Increased from 3 for self-correction room

    # ── Reliability & Verification (Phase 7) ───────────────
    verification_enabled: bool = True


# ── Singleton accessor ──────────────────────────────────────
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
