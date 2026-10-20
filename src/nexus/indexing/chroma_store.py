"""
nexus.indexing.chroma_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~
ChromaDB persistence layer — the "db_manager.py" the README promised but never existed.

Responsibilities:
  1. Create/get ChromaDB collections with cosine similarity metric
  2. Upsert chunks (idempotent — reinserting the same chunk_id updates, doesn't duplicate)
  3. Tier-aware retention cleanup
  4. Query interface used by the retrieval layer
  5. Collection stats for monitoring

Why ChromaDB?
  - Embedded (no separate server process)
  - PersistentClient writes to disk automatically
  - Supports metadata filtering ($gte, $lt, $eq, $and) for date/tier/source filters
  - cosine similarity is appropriate for unit-normalised embeddings

The collection uses cosine distance. ChromaDB stores distance (lower=more similar),
but our retrieval layer converts to similarity scores (higher=more similar) for
consistency with the reranker.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from nexus.ingestion.models import ArticleChunk, SourceTier

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    Manages all interactions with ChromaDB for a single collection.
    Thread-safe: ChromaDB's PersistentClient handles concurrent reads.
    (Writes should be serialised — the ingestion scheduler handles this.)
    """

    def __init__(self, chroma_path: str, collection_name: str) -> None:
        """
        Args:
            chroma_path: Filesystem path where ChromaDB stores its data.
            collection_name: Name of the ChromaDB collection.
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name

        # Disable ChromaDB telemetry (prevents connection errors in offline envs)
        self._client = chromadb.PersistentClient(
            path=chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Create collection with cosine similarity metric
        # get_or_create_collection is idempotent — safe to call on every startup
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine distance metric
        )
        logger.info(
            "ChromaStore ready: collection='%s', path='%s', count=%d",
            collection_name,
            chroma_path,
            self._collection.count(),
        )

    # ── Write operations ──────────────────────────────────────────

    def upsert_chunks(self, chunks: list[ArticleChunk], embeddings: list[list[float]]) -> int:
        """
        Upsert chunk documents and their pre-computed embeddings into ChromaDB.

        Why upsert (not add)?
          upsert() updates existing records with the same ID and inserts new ones.
          This makes ingestion idempotent: re-running the pipeline with the same
          articles doesn't create duplicates.

        Args:
            chunks: List of ArticleChunk to store.
            embeddings: Pre-computed embedding vectors (same length as chunks).

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        assert len(chunks) == len(embeddings), "chunks and embeddings must have same length"

        batch_size = 512  # ChromaDB handles batches efficiently up to ~1000

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            self._collection.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                documents=[c.text for c in batch_chunks],
                embeddings=batch_embeddings,
                metadatas=[c.to_chroma_metadata() for c in batch_chunks],
            )
            logger.debug("Upserted batch %d/%d", i // batch_size + 1, (len(chunks) - 1) // batch_size + 1)

        logger.info("Upserted %d chunks into '%s'", len(chunks), self.collection_name)
        return len(chunks)

    def run_retention_cleanup(
        self,
        retention_days_tier1: int = 180,
        retention_days_tier2: int = 365,
        retention_days_tier3: int = 14,
    ) -> dict[str, int]:
        """
        Delete chunks older than their tier's retention window.

        Tier-aware retention:
          - Tier 1 (official): 180 days — government announcements stay relevant
          - Tier 2 (fact-checkers): 365 days — fact-check articles are evergreen
          - Tier 3 (news): 14 days — news becomes stale quickly

        Args:
            retention_days_tier1: Days to keep Tier 1 articles.
            retention_days_tier2: Days to keep Tier 2 articles.
            retention_days_tier3: Days to keep Tier 3 articles.

        Returns:
            Dict of {tier: n_deleted} counts.
        """
        now = datetime.now(tz=timezone.utc)
        deleted_counts = {}

        tier_config = {
            SourceTier.OFFICIAL: retention_days_tier1,
            SourceTier.FACT_CHECKER: retention_days_tier2,
            SourceTier.NEWS: retention_days_tier3,
        }

        for tier, days in tier_config.items():
            cutoff_ts = int((now - timedelta(days=days)).timestamp())
            self._collection.delete(
                where={
                    "$and": [
                        {"tier": {"$eq": int(tier)}},
                        {"published_ts": {"$lt": cutoff_ts}},
                    ]
                }
            )
            deleted_counts[f"tier{int(tier)}"] = 0  # ChromaDB doesn't return delete count
            logger.info(
                "Retention cleanup: tier=%d, cutoff=%s (ts=%d)",
                tier,
                (now - timedelta(days=days)).isoformat()[:10],
                cutoff_ts,
            )

        return deleted_counts

    # ── Read operations ───────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 20,
        where: dict | None = None,
    ) -> dict[str, Any]:
        """
        Perform a vector similarity search.

        Args:
            query_embedding: Query vector (must match embedding model dimension).
            n_results: Number of results to return.
            where: Optional ChromaDB metadata filter dict.

        Returns:
            ChromaDB query result dict with keys: ids, documents, metadatas, distances.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(self._collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return self._collection.query(**kwargs)

    def get_all_documents(self) -> dict[str, Any]:
        """
        Retrieve all stored documents and metadata.
        Used for building the BM25 index at startup.

        Returns:
            Dict with keys: ids, documents, metadatas.
        """
        count = self._collection.count()
        if count == 0:
            return {"ids": [], "documents": [], "metadatas": []}

        return self._collection.get(
            limit=count,
            include=["documents", "metadatas"],
        )

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._collection.count()

    def get_collection_stats(self) -> dict[str, Any]:
        """Return summary statistics about the collection."""
        count = self._collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "chroma_path": self.chroma_path,
        }
