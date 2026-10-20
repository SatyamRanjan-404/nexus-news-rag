"""
nexus.ingestion.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~
Top-level ingestion orchestrator — runs the full pipeline in one call.

Steps:
  1. Load sources from news_sources.json
  2. Fetch all feeds concurrently (feedparser + trafilatura)
  3. Deduplicate (SimHash near-duplicate detection)
  4. Chunk text (recursive or semantic based on config)
  5. Embed chunks
  6. Upsert into ChromaDB (idempotent)
  7. Rebuild BM25 index
  8. Run retention cleanup (tier-aware)
  9. Emit IngestionStats

This replaces the original schedular.py / run_full_pipeline() function
with a properly structured, logged, and monitored pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from nexus.chunking.factory import ChunkerFactory
from nexus.config.settings import Settings
from nexus.embeddings.model import get_embedding_model
from nexus.indexing.bm25_index import BM25Index
from nexus.indexing.chroma_store import ChromaStore
from nexus.ingestion.dedup import deduplicate_articles
from nexus.ingestion.fetcher import fetch_all_feeds, load_sources
from nexus.ingestion.models import IngestionStats

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    settings: Settings,
    chroma_store: ChromaStore,
    bm25_index_ref: list,   # mutable container: bm25_index_ref[0] = BM25Index
    sources_path: str | Path = "news_sources.json",
) -> IngestionStats:
    """
    Run the complete ingestion pipeline.

    Args:
        settings: Application configuration.
        chroma_store: ChromaStore instance to upsert into.
        bm25_index_ref: Single-element list holding the current BM25Index.
                        After ingestion, bm25_index_ref[0] is replaced with
                        the freshly rebuilt index. Using a list so the caller's
                        reference is updated in-place (Python doesn't have pointers).
        sources_path: Path to news_sources.json.

    Returns:
        IngestionStats with metrics from this pipeline run.
    """
    stats = IngestionStats()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE STARTED")
    logger.info("=" * 60)

    # Step 1: Load sources
    sources = load_sources(sources_path)
    stats.feed_total = len(sources)

    if not sources:
        logger.error("No enabled sources found in %s", sources_path)
        stats.errors.append("No enabled sources found")
        return stats

    # Step 2: Fetch feeds
    logger.info("Step 1/6: Fetching %d feeds...", len(sources))
    articles, feed_results = fetch_all_feeds(
        sources=sources,
        max_workers=settings.fetch_workers,
    )
    stats.feed_ok = sum(1 for v in feed_results.values() if v)
    stats.feed_failed = sum(1 for v in feed_results.values() if not v)
    stats.articles_fetched = len(articles)
    stats.articles_full_text = sum(1 for a in articles if a.extraction_method == "full_text")
    stats.articles_summary_fallback = sum(1 for a in articles if a.extraction_method == "summary_fallback")

    logger.info(
        "Fetch complete: %d articles, %d full-text (%.0f%%), %d summary fallback",
        stats.articles_fetched,
        stats.articles_full_text,
        100 * stats.articles_full_text / max(stats.articles_fetched, 1),
        stats.articles_summary_fallback,
    )

    # Step 3: Deduplicate
    logger.info("Step 2/6: Deduplicating articles...")
    deduped_articles, n_dropped = deduplicate_articles(articles)
    stats.articles_deduped = n_dropped
    stats.articles_indexed = len(deduped_articles)

    # Per-tier counts
    stats.tier1_count = sum(1 for a in deduped_articles if a.tier == 1)
    stats.tier2_count = sum(1 for a in deduped_articles if a.tier == 2)
    stats.tier3_count = sum(1 for a in deduped_articles if a.tier == 3)

    # Step 4: Chunk
    logger.info("Step 3/6: Chunking %d articles...", len(deduped_articles))
    embedding_model = get_embedding_model(settings.embedding_model)

    chunker = ChunkerFactory.create(settings, embedding_model=embedding_model)
    all_chunks = chunker.chunk_many(deduped_articles)
    stats.chunks_created = len(all_chunks)

    logger.info("Chunking complete: %d chunks from %d articles", len(all_chunks), len(deduped_articles))

    # Step 5: Embed
    logger.info("Step 4/6: Embedding %d chunks...", len(all_chunks))
    chunk_texts = [c.text for c in all_chunks]

    batch_size = settings.embedding_batch_size
    all_embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i : i + batch_size]
        batch_embs = embedding_model.embed_documents(batch)
        all_embeddings.extend(batch_embs)
        if (i // batch_size + 1) % 10 == 0:
            logger.debug("Embedded batch %d/%d", i // batch_size + 1, len(chunk_texts) // batch_size + 1)

    logger.info("Embedding complete: %d vectors", len(all_embeddings))

    # Step 6: Upsert into ChromaDB
    logger.info("Step 5/6: Upserting into ChromaDB...")
    upserted = chroma_store.upsert_chunks(all_chunks, all_embeddings)
    stats.chunks_upserted = upserted

    # Rebuild BM25 index from updated ChromaDB
    logger.info("Step 5b/6: Rebuilding BM25 index...")
    new_bm25 = BM25Index.build_from_chroma(chroma_store)
    bm25_index_ref[0] = new_bm25  # update in-place for live reference

    # Step 7: Retention cleanup
    logger.info("Step 6/6: Running retention cleanup...")
    chroma_store.run_retention_cleanup(
        retention_days_tier1=settings.retention_days_tier1,
        retention_days_tier2=settings.retention_days_tier2,
        retention_days_tier3=settings.retention_days_tier3,
    )

    stats.duration_seconds = time.time() - start_time

    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE COMPLETE in %.1fs", stats.duration_seconds)
    logger.info("  Feed success rate: %d/%d", stats.feed_ok, stats.feed_total)
    logger.info("  Articles: %d fetched → %d after dedup → %d chunks",
                stats.articles_fetched, stats.articles_indexed, stats.chunks_created)
    logger.info("  Full-text extraction: %.0f%%",
                100 * stats.articles_full_text / max(stats.articles_fetched, 1))
    logger.info("  ChromaDB total: %d chunks", chroma_store.count())
    logger.info("=" * 60)

    return stats
