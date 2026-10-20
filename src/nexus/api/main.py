"""
nexus.api.main
~~~~~~~~~~~~~~
FastAPI application entry point.

Endpoints:
  POST  /v1/ask          — RAG question answering (streaming or blocking)
  POST  /v1/check-claim  — News claim checker
  GET   /v1/sources      — List all configured news sources
  POST  /v1/ingest       — Trigger manual ingestion (API-key protected)
  GET   /health          — Liveness probe (always fast)
  GET   /ready           — Readiness probe (checks DB is populated)

Design:
  - All heavy objects (ChromaStore, BM25Index, embedder, reranker) are
    created ONCE at startup via FastAPI's lifespan context manager
  - Rate limiting via slowapi (per-IP)
  - Structured JSON logging via structlog
  - CORS configured from settings
  - All errors return {error, detail, request_id} for consistency
  - Request IDs flow through all log lines for tracing
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from nexus.api.models import (
    AskRequest,
    AskResponse,
    Citation,
    ClaimCheckAPIRequest,
    HealthResponse,
    IngestResponse,
    ReadyResponse,
    SourceInfo,
    SourcesResponse,
)
from nexus.claims.checker import ClaimChecker
from nexus.config.settings import get_settings
from nexus.embeddings.model import get_embedding_model
from nexus.generation.rag_chain import RAGChain
from nexus.indexing.bm25_index import BM25Index
from nexus.indexing.chroma_store import ChromaStore
from nexus.ingestion.fetcher import load_sources
from nexus.ingestion.pipeline import run_ingestion_pipeline
from nexus.rerank.cross_encoder import CrossEncoderReranker
from nexus.retrieval.hybrid import HybridRetriever
from nexus.retrieval.query_rewriter import QueryRewriter

logger = structlog.get_logger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Application state (shared across requests) ────────────────────

class AppState:
    """Container for all heavy objects initialised at startup."""
    chroma_store: ChromaStore | None = None
    bm25_index_ref: list[BM25Index | None] = [None]
    retriever: HybridRetriever | None = None
    reranker: CrossEncoderReranker | None = None
    rag_chain: RAGChain | None = None
    claim_checker: ClaimChecker | None = None
    query_rewriter: QueryRewriter | None = None
    last_ingestion: str | None = None
    ingestion_running: bool = False


app_state = AppState()

# ── Lifespan (startup / shutdown) ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise all heavy objects (models, DB connections, etc.)
    Shutdown: graceful teardown.
    """
    settings = get_settings()

    logger.info("Starting Nexus API", version="0.2.0")

    # Initialise ChromaDB
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    app_state.chroma_store = ChromaStore(
        chroma_path=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )

    # Load embedding model (cached)
    embedding_model = get_embedding_model(settings.embedding_model)

    # Build BM25 index from existing ChromaDB data
    app_state.bm25_index_ref[0] = BM25Index.build_from_chroma(app_state.chroma_store)

    # Initialise retriever
    app_state.retriever = HybridRetriever(
        chroma_store=app_state.chroma_store,
        bm25_index=app_state.bm25_index_ref[0],
        embedding_model=embedding_model,
        rrf_k=settings.rrf_k,
    )

    # Initialise reranker
    app_state.reranker = CrossEncoderReranker()

    # Initialise RAG chain
    app_state.rag_chain = RAGChain(
        groq_api_key=settings.groq_api_key,
        model=settings.generator_model,
        temperature=settings.generator_temperature,
    )

    # Initialise claim checker
    app_state.claim_checker = ClaimChecker(
        retriever=app_state.retriever,
        reranker=app_state.reranker,
        groq_api_key=settings.groq_api_key,
        model=settings.generator_model,
    )

    # Initialise query rewriter
    app_state.query_rewriter = QueryRewriter(
        groq_api_key=settings.groq_api_key,
        model=settings.judge_model,
    )

    # Trigger initial ingestion in background thread if DB is empty
    if app_state.chroma_store.count() == 0:
        logger.info("ChromaDB empty — triggering initial ingestion in background")
        threading.Thread(target=_run_ingestion_sync, daemon=True).start()
    else:
        logger.info(
            "ChromaDB has existing data",
            chunks=app_state.chroma_store.count(),
        )

    # Schedule periodic ingestion
    _schedule_ingestion(settings.ingest_interval_minutes)

    yield  # Application runs here

    logger.info("Nexus API shutting down")


def _run_ingestion_sync():
    """Run the ingestion pipeline in a background thread."""
    if app_state.ingestion_running:
        logger.info("Ingestion already running — skipping")
        return

    app_state.ingestion_running = True
    try:
        settings = get_settings()
        stats = run_ingestion_pipeline(
            settings=settings,
            chroma_store=app_state.chroma_store,
            bm25_index_ref=app_state.bm25_index_ref,
        )
        app_state.last_ingestion = datetime.now(tz=timezone.utc).isoformat()

        # Update retriever with fresh BM25 index
        if app_state.bm25_index_ref[0] is not None and app_state.retriever is not None:
            app_state.retriever._bm25 = app_state.bm25_index_ref[0]

        logger.info("Ingestion complete", stats=stats.to_dict())
    except Exception as e:
        logger.error("Ingestion pipeline failed", error=str(e))
    finally:
        app_state.ingestion_running = False


def _schedule_ingestion(interval_minutes: int):
    """Schedule periodic ingestion using threading.Timer (simple, no APScheduler needed)."""
    def run_and_reschedule():
        _run_ingestion_sync()
        _schedule_ingestion(interval_minutes)

    timer = threading.Timer(interval_minutes * 60, run_and_reschedule)
    timer.daemon = True
    timer.start()


# ── FastAPI app ───────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="Nexus News RAG API",
    description=(
        "AI-powered news retrieval and claim checking API. "
        "Queries are grounded in a rolling window of recent news articles. "
        "Claim checker verdicts reflect what retrieved articles say, not absolute truth."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request ID logging ────────────────────────────────

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Helper: API key auth ─────────────────────────────────────────

def verify_ingest_key(x_api_key: str = Header(...)):
    if x_api_key != settings.ingest_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ingest API key",
        )


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health():
    """
    Liveness probe — always returns quickly.
    If this endpoint is down, the service is down.
    """
    return HealthResponse(status="ok", version="0.2.0")


@app.get("/ready", response_model=ReadyResponse, tags=["monitoring"])
async def ready():
    """
    Readiness probe — checks if the DB is populated and ready to serve queries.
    Returns ready=false during initial ingestion on a cold start.
    """
    chunk_count = app_state.chroma_store.count() if app_state.chroma_store else 0
    bm25_count = len(app_state.bm25_index_ref[0]) if app_state.bm25_index_ref[0] else 0
    is_ready = chunk_count > 0

    return ReadyResponse(
        ready=is_ready,
        chroma_chunks=chunk_count,
        bm25_docs=bm25_count,
        last_ingestion=app_state.last_ingestion,
        message="Ready" if is_ready else "Initialising — ingestion in progress. Please wait.",
    )


@app.get("/v1/sources", response_model=SourcesResponse, tags=["data"])
async def get_sources():
    """List all configured news sources with their tier and category."""
    sources_data = load_sources("news_sources.json")
    return SourcesResponse(
        sources=[
            SourceInfo(
                name=s["name"],
                category=s.get("category", "General"),
                tier=s.get("tier", 3),
                enabled=s.get("enabled", True),
                url=s["url"],
            )
            for s in sources_data
        ],
        total=len(sources_data),
    )


@app.post("/v1/ask", tags=["rag"])
@limiter.limit(settings.rate_limit_ask)
async def ask(request: Request, body: AskRequest):
    """
    RAG question answering — retrieves relevant news and generates a grounded answer.

    - Rewrites follow-up queries if conversation history is provided
    - Uses hybrid dense+BM25 retrieval with cross-encoder reranking
    - Streams the response by default
    - Returns citations with each answer
    """
    if not app_state.retriever or not app_state.rag_chain:
        raise HTTPException(status_code=503, detail="Service not ready — ingestion in progress")

    # Rewrite query if follow-up
    rewritten_query = body.query
    if body.conversation_history and app_state.query_rewriter:
        rewritten_query = app_state.query_rewriter.rewrite(
            query=body.query,
            conversation_history=body.conversation_history,
        )

    # Build optional metadata filter
    where_clause = None
    if body.source_filter and body.source_filter != "All Feeds":
        where_clause = {"source": {"$eq": body.source_filter}}

    # Retrieve
    candidates = app_state.retriever.retrieve(
        query=rewritten_query,
        n_results=settings.rerank_candidate_k,
        where=where_clause,
    )

    # Rerank
    reranked = app_state.reranker.rerank(
        query=rewritten_query,
        candidates=candidates,
        top_k=settings.final_k,
    )

    if body.stream:
        # Streaming response
        stream, citations = app_state.rag_chain.stream(rewritten_query, reranked)

        if stream is None:
            return JSONResponse(
                content=AskResponse(
                    answer=None,
                    citations=[],
                    rewritten_query=rewritten_query if rewritten_query != body.query else None,
                ).model_dump()
            )

        async def generate():
            # Stream the answer tokens
            for chunk in stream:
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            # After the stream, send citations
            yield f"data: {json.dumps({'citations': [c for c in citations], 'done': True})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    else:
        # Blocking response
        answer, citations = app_state.rag_chain.invoke(rewritten_query, reranked)
        return AskResponse(
            answer=answer,
            citations=[Citation(**c) for c in citations],
            rewritten_query=rewritten_query if rewritten_query != body.query else None,
        )


@app.post("/v1/check-claim", tags=["claim-checker"])
@limiter.limit(settings.rate_limit_claim)
async def check_claim(request: Request, body: ClaimCheckAPIRequest):
    """
    News Claim Checker — verifies a claim against the retrieved news corpus.

    Returns one of: SUPPORTED, CONTRADICTED, or NOT_ENOUGH_EVIDENCE.

    IMPORTANT: The verdict reflects what retrieved news articles say,
    not absolute truth. A SUPPORTED verdict means the claim is supported
    by articles in our rolling news index, not that it is definitely true.
    """
    if not app_state.claim_checker:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        verdict, verified_count = app_state.claim_checker.check(
            claim=body.claim,
            rerank_candidate_k=settings.rerank_candidate_k,
            final_k=settings.final_k,
            tier_filter=body.tier_filter,
        )

        return {
            "claim": body.claim,
            "result": verdict.model_dump(),
            "verified_evidence_count": verified_count,
            "disclaimer": (
                "This verdict is based on articles retrieved from our news index "
                "(a rolling window of recent articles). It does not represent absolute truth. "
                "SUPPORTED means the retrieved articles support this claim."
            ),
        }
    except Exception as e:
        logger.error("Claim check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Claim check failed: {str(e)}")


@app.post(
    "/v1/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_ingest_key)],
    tags=["admin"],
)
async def trigger_ingest(request: Request):
    """
    Trigger a manual ingestion cycle (API-key protected).
    The ingestion runs in the background — returns immediately.
    Check /ready to monitor progress.
    """
    if app_state.ingestion_running:
        return IngestResponse(status="already_running", message="Ingestion is already in progress")

    threading.Thread(target=_run_ingestion_sync, daemon=True).start()
    return IngestResponse(status="started", message="Ingestion started in background. Check /ready for status.")
