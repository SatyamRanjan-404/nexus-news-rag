"""
nexus.config.settings
~~~~~~~~~~~~~~~~~~~~~
Single source of truth for all configuration.

All values come from environment variables (or a .env file via python-dotenv).
Import get_settings() anywhere in the codebase — it is cached after the first call.

Usage:
    from nexus.config.settings import get_settings
    s = get_settings()
    print(s.groq_api_key)
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration in one place.
    Values are read from environment variables (case-insensitive).
    If a .env file exists in the working directory, it is loaded automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── API Keys ─────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key (required)")
    ingest_api_key: str = Field(
        default="change_me", description="Secret key for POST /v1/ingest"
    )

    # ── LLM ──────────────────────────────────────────────────────
    generator_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model for answer generation",
    )
    judge_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model for LLM-as-judge evaluation (different from generator to reduce self-preference bias)",
    )
    generator_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    claim_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=0.0,
        description="Claim checker must use temp=0 for reproducibility",
    )

    # ── Embeddings ───────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model name",
    )
    embedding_batch_size: int = Field(default=64, ge=1)

    # ── Chunking ─────────────────────────────────────────────────
    chunking_strategy: Literal["recursive", "semantic"] = Field(
        default="recursive",
        description="'recursive' uses RecursiveCharacterTextSplitter; 'semantic' splits on sentence embedding similarity",
    )
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0)

    # ── Retrieval ────────────────────────────────────────────────
    rerank_candidate_k: int = Field(
        default=20,
        description="Number of candidates fetched before cross-encoder reranking",
    )
    final_k: int = Field(
        default=5,
        description="Number of chunks passed to the LLM after reranking",
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant k. Standard value is 60. Higher k reduces the impact of rank differences.",
    )

    # ── ChromaDB ─────────────────────────────────────────────────
    chroma_path: str = Field(default="./data/chroma_db")
    chroma_collection: str = Field(default="news_collection")
    chroma_eval_collection: str = Field(default="news_eval")

    # ── Ingestion ────────────────────────────────────────────────
    ingest_interval_minutes: int = Field(default=15, ge=5)
    retention_days_tier3: int = Field(default=14, description="News articles (Tier 3)")
    retention_days_tier1: int = Field(default=180, description="Official sources (Tier 1)")
    retention_days_tier2: int = Field(default=365, description="Fact-checkers (Tier 2)")
    fetch_timeout_seconds: int = Field(default=5, ge=1, le=30)
    fetch_workers: int = Field(default=10, ge=1, le=50)

    # ── Rate Limiting ────────────────────────────────────────────
    rate_limit_ask: str = Field(default="20/minute")
    rate_limit_claim: str = Field(default="10/minute")

    # ── API Server ───────────────────────────────────────────────
    cors_origins: str = Field(default="http://localhost:8501")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(default="console")

    # ── Evaluation ───────────────────────────────────────────────
    eval_snapshot_path: str = Field(
        default="./data/snapshots",
        description="Directory where frozen corpus snapshots are stored",
    )
    citation_fuzzy_threshold: float = Field(
        default=0.85,
        description="Minimum fuzzy match score for a model-cited quote to be accepted",
    )

    @field_validator("cors_origins")
    @classmethod
    def parse_cors(cls, v: str) -> str:
        """Accepts comma-separated origins."""
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        """Returns CORS origins as a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the cached Settings instance.
    Cached so the .env file is only read once per process.
    Call get_settings.cache_clear() in tests to reset.
    """
    return Settings()  # type: ignore[call-arg]
