"""
nexus.api.models
~~~~~~~~~~~~~~~~
Pydantic request/response models for the FastAPI API.

These models serve multiple purposes:
  1. Input validation — FastAPI validates request bodies against these
  2. OpenAPI docs — FastAPI generates Swagger/ReDoc docs from them
  3. Type safety — mypy checks usage against these models
  4. Response consistency — errors always have the same structure

All models use clear field descriptions so the auto-generated API docs
are useful without reading the source code.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Common fields ─────────────────────────────────────────────────


class RequestMeta(BaseModel):
    """Common metadata for API responses."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request ID for tracing",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
    )


class ErrorResponse(BaseModel):
    """Standard error response shape — all errors use this format."""

    error: str = Field(description="Error type/code")
    detail: str = Field(description="Human-readable error message")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ── Ask endpoint ──────────────────────────────────────────────────


class AskRequest(BaseModel):
    """POST /v1/ask request body."""

    query: str = Field(
        min_length=3,
        max_length=500,
        description="Question about recent news",
        examples=["What are the latest updates on the India-China border?"],
    )
    source_filter: str | None = Field(
        default=None,
        description="Restrict to a specific source name (must match a name in /v1/sources)",
    )
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior conversation turns for query rewriting. "
        "Each item: {role: 'user'|'assistant', content: str}",
        max_length=20,
    )
    stream: bool = Field(
        default=True,
        description="If true, streams the response as SSE. If false, returns full response.",
    )


class Citation(BaseModel):
    """A single source citation in the answer."""

    title: str
    link: str
    source: str
    published: str


class AskResponse(BaseModel):
    """POST /v1/ask response body (non-streaming mode)."""

    answer: str | None = Field(
        description="Generated answer, or null if no relevant context was found"
    )
    citations: list[Citation] = Field(default_factory=list)
    rewritten_query: str | None = Field(
        default=None,
        description="The query after rewriting (if conversation history was provided)",
    )
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ── Claim check endpoint ──────────────────────────────────────────


class ClaimCheckAPIRequest(BaseModel):
    """POST /v1/check-claim request body."""

    claim: str = Field(
        min_length=10,
        max_length=2000,
        description="The claim, rumour, or forward to verify",
        examples=["The government has announced a 20% reduction in income tax."],
    )
    tier_filter: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="Restrict evidence to: 1=official sources, 2=fact-checkers, 3=news only",
    )


# ── Sources endpoint ──────────────────────────────────────────────


class SourceInfo(BaseModel):
    """One entry in the GET /v1/sources response."""

    name: str
    category: str
    tier: int
    enabled: bool
    url: str


class SourcesResponse(BaseModel):
    """GET /v1/sources response."""

    sources: list[SourceInfo]
    total: int


# ── Ingest endpoint ───────────────────────────────────────────────


class IngestResponse(BaseModel):
    """POST /v1/ingest response."""

    status: str  # "started" or "already_running"
    message: str


# ── Health endpoints ──────────────────────────────────────────────


class HealthResponse(BaseModel):
    """GET /health response — fast liveness check."""

    status: str  # "ok" or "degraded"
    version: str


class ReadyResponse(BaseModel):
    """GET /ready response — readiness check (is the DB populated?)."""

    ready: bool
    chroma_chunks: int
    bm25_docs: int
    last_ingestion: str | None
    message: str
