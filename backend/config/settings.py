"""
Centralized application settings loaded from environment variables / .env file.
Uses Pydantic Settings for validation, type coercion, and defaults.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from pathlib import Path
from typing import List, Optional

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import make_url


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
    debug: bool = False
    log_level: str = "INFO"
    auth_secret: str = "change-this-auth-secret-in-production"
    auth_frontend_url: str = "http://localhost:8000"
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    google_client_id: Optional[str] = None
    google_client_secret: Optional[SecretStr] = None
    github_client_id: Optional[str] = None
    github_client_secret: Optional[SecretStr] = None
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5500",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("supabase_url", mode="before")
    @classmethod
    def clean_supabase_url(cls, v):
        if isinstance(v, str) and v.strip():
            url = v.strip().rstrip("/")
            if url.endswith("/rest/v1"):
                url = url[:-8].rstrip("/")
            return url
        return v

    # ── Database / PostgreSQL / Supabase ────────────────────
    database_provider: str = "postgres"  # "postgres" or "sqlite"
    database_url_override: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "database_url",
            "supabase_db_url",
            "postgres_url",
            "supabase_database_url",
        ),
    )
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password_change_me"
    postgres_db: str = "advanced_rag"
    postgres_ssl: Optional[str] = "require"

    @property
    def database_url(self) -> str:
        """Async SQLAlchemy connection URL (asyncpg for PostgreSQL)."""
        if (
            self.database_provider.lower() == "sqlite"
            or (self.app_env.lower() == "test" and not self.database_url_override)
        ):
            return "sqlite+aiosqlite:///./advanced_rag.db"

        if self.database_url_override:
            url_str = self.database_url_override.strip()
            if url_str.startswith("postgres://"):
                url_str = "postgresql+asyncpg://" + url_str[len("postgres://") :]
            elif url_str.startswith("postgresql://"):
                url_str = "postgresql+asyncpg://" + url_str[len("postgresql://") :]
            elif url_str.startswith("sqlite"):
                return url_str

            try:
                u = make_url(url_str)
                query = dict(u.query)
                if "sslmode" in query:
                    query.pop("sslmode")
                u = u._replace(query=query)
                return u.render_as_string(hide_password=False)
            except Exception:
                return url_str

        user = urllib.parse.quote_plus(self.postgres_user)
        password = urllib.parse.quote_plus(self.postgres_password)
        return (
            f"postgresql+asyncpg://{user}:{password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """Synchronous SQLAlchemy connection URL (psycopg2 for PostgreSQL)."""
        if (
            self.database_provider.lower() == "sqlite"
            or (self.app_env.lower() == "test" and not self.database_url_override)
        ):
            return "sqlite:///./advanced_rag.db"

        if self.database_url_override:
            url_str = self.database_url_override.strip()
            if url_str.startswith("postgres://"):
                url_str = "postgresql+psycopg2://" + url_str[len("postgres://") :]
            elif url_str.startswith("postgresql+asyncpg://"):
                url_str = "postgresql+psycopg2://" + url_str[len("postgresql+asyncpg://") :]
            elif url_str.startswith("postgresql://"):
                url_str = "postgresql+psycopg2://" + url_str[len("postgresql://") :]
            elif url_str.startswith("sqlite"):
                return url_str
            return url_str

        user = urllib.parse.quote_plus(self.postgres_user)
        password = urllib.parse.quote_plus(self.postgres_password)
        base = f"postgresql+psycopg2://{user}:{password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        if self.postgres_ssl and self.postgres_ssl.lower() not in ("disable", "false", "0", "no"):
            base += f"?sslmode={self.postgres_ssl}"
        return base

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
    openai_api_key: Optional[SecretStr] = None
    google_api_key: Optional[SecretStr] = None
    groq_api_key: Optional[SecretStr] = None

    @model_validator(mode="after")
    def _warn_placeholder_keys(self) -> "Settings":
        """Emit a startup warning if API keys look like placeholders."""
        _log = logging.getLogger("backend.config")
        if self.app_env.lower() in {"production", "prod"}:
            if self.debug:
                raise ValueError("DEBUG must be false in production")
            if self.auth_secret.strip().lower() in {
                "change-this-auth-secret-in-production",
                "replace-with-a-long-random-secret",
                "changeme",
                "change-me",
            }:
                raise ValueError("AUTH_SECRET must be configured in production")

        placeholders = {"your-openai-api-key-here", "your-google-api-key-here", "", "changeme"}
        if self.openai_api_key and self.openai_api_key.get_secret_value().lower() in placeholders:
            _log.warning("OPENAI_API_KEY looks like a placeholder — OpenAI features will fail")
        if self.google_api_key and self.google_api_key.get_secret_value().lower() in placeholders:
            _log.warning("GOOGLE_API_KEY looks like a placeholder — Gemini features will fail")
        if self.groq_api_key and self.groq_api_key.get_secret_value().lower() in placeholders:
            _log.warning("GROQ_API_KEY looks like a placeholder — Groq features will fail")
        return self

    # ── Ingestion ───────────────────────────────────────────
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    allowed_extensions: List[str] = [
        ".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls", ".docx",
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
    retrieval_top_k: int = 7
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
    reranker_top_k: int = 7  # Final top-K after reranking

    # ── Query Processing (Phase 2) ─────────────────────────
    query_rewriting_enabled: bool = True
    query_expansion_enabled: bool = False  # Off by default, uses extra LLM calls
    query_decomposition_enabled: bool = False  # For Phase 6 agentic

    # ── Context Compression (Phase 2) ──────────────────────
    context_compression_enabled: bool = True
    max_context_tokens: int = 6000  # Max tokens sent to LLM

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
    agent_token_budget: int = 50000  # Max estimated tokens across all LLM calls per query

    # ── Reliability & Verification (Phase 7) ───────────────
    verification_enabled: bool = True

    # ── Usage Limits ───────────────────────────────────────
    usage_window_days: int = 30
    usage_max_tokens: int = 0  # 0 means unlimited
    usage_max_cost_usd: float = 0.0  # 0 means unlimited


# ── Singleton accessor ──────────────────────────────────────
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
