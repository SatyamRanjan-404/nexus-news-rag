"""
nexus.retrieval.hybrid
~~~~~~~~~~~~~~~~~~~~~~~
Hybrid retrieval: dense (ChromaDB) + BM25, fused via Reciprocal Rank Fusion (RRF).

What is RRF?
  Reciprocal Rank Fusion is a score combination method that works on RANKS
  (positions in each result list) rather than raw scores. This is important
  because BM25 scores and cosine similarity scores are on completely
  different scales and cannot be meaningfully added or averaged.

  RRF formula:
    rrf_score(document) = Σ 1 / (k + rank_i)

  Where:
    - rank_i: the rank of this document in the i-th retrieval list (1-indexed)
    - k: a constant (default 60). Higher k reduces the impact of rank
         differences between lists, making it more "fair" to all retrievers.
    - The sum is over all retrieval lists (dense + BM25)

  Example: if a document is rank #1 in dense (1/61 = 0.0164) and
  rank #3 in BM25 (1/63 = 0.0159), its RRF score = 0.0323.
  A document only in dense at rank #5 gets 1/65 = 0.0154.

  Documents that appear in both lists get a BONUS — they are more likely
  to be genuinely relevant.

Why RRF over linear interpolation?
  Linear interpolation requires tuning weights AND normalising scores.
  BM25 scores vary wildly by query (from 0 to 50+ for popular terms).
  Cosine similarities are in [-1, 1]. Mixing them directly without
  normalisation produces inconsistent results. RRF is scale-invariant:
  it only cares about ranks, so no normalisation is needed.
  This is why RRF is used in production retrieval systems (e.g., Elastic's KNN).

Key parameters (from Phase 7 experiments):
  - rrf_k: 60 (standard), tested 30/60/90
  - dense_k: number of dense candidates (e.g. 20)
  - bm25_k: number of BM25 candidates (e.g. 20)
  - final_k: number of results AFTER reranking (e.g. 5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """
    A single retrieved text chunk with its metadata and relevance scores.
    This is the object passed between retrieval → reranking → generation.
    """

    chunk_id: str
    text: str
    title: str
    url: str
    source_name: str
    category: str
    tier: int
    published_str: str
    published_ts: int
    extraction_method: str

    # Retrieval scores (filled in by the retrieval pipeline)
    dense_score: float = 0.0    # cosine similarity (0-1, higher = more similar)
    bm25_score: float = 0.0     # raw BM25 score
    rrf_score: float = 0.0      # RRF fusion score (higher = better)
    rerank_score: float = 0.0   # cross-encoder score (filled in by reranker)

    @classmethod
    def from_chroma_result(
        cls,
        chunk_id: str,
        document: str,
        metadata: dict,
        distance: float,
    ) -> "RetrievedChunk":
        """
        Construct from a ChromaDB query result row.

        ChromaDB returns 'distance' (lower = more similar for cosine metric).
        We convert: similarity = 1 - distance.
        """
        return cls(
            chunk_id=chunk_id,
            text=document,
            title=metadata.get("title", ""),
            url=metadata.get("link", ""),
            source_name=metadata.get("source", ""),
            category=metadata.get("category", ""),
            tier=metadata.get("tier", 3),
            published_str=metadata.get("published", ""),
            published_ts=metadata.get("published_ts", 0),
            extraction_method=metadata.get("extraction_method", "unknown"),
            dense_score=1.0 - float(distance),  # convert distance to similarity
        )


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    bm25_results: list[tuple[str, float]],
    all_chunks_by_id: dict[str, RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Fuse dense and BM25 retrieval results using Reciprocal Rank Fusion.

    Args:
        dense_results: Ordered list of chunks from dense retrieval (rank 0 = best).
        bm25_results: List of (chunk_id, bm25_score) from BM25 (ordered best-first).
        all_chunks_by_id: Dict mapping chunk_id → RetrievedChunk (pre-populated
                           from ChromaDB for any chunk that BM25 found but dense didn't).
        k: RRF constant (default 60).

    Returns:
        List of RetrievedChunk sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}

    # Assign RRF scores from dense ranking
    for rank, chunk in enumerate(dense_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)

    # Assign RRF scores from BM25 ranking
    for rank, (chunk_id, bm25_score) in enumerate(bm25_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    # Collect all unique chunk IDs across both lists
    all_ids = set(rrf_scores.keys())

    # Build final result list
    result_chunks: list[RetrievedChunk] = []
    seen_ids = set()

    for chunk_id in sorted(all_ids, key=lambda cid: rrf_scores[cid], reverse=True):
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        # Find the chunk object
        chunk = all_chunks_by_id.get(chunk_id)
        if chunk is None:
            continue  # chunk only in BM25 results but not in ChromaDB lookup — skip

        chunk.rrf_score = rrf_scores[chunk_id]
        result_chunks.append(chunk)

    return result_chunks


class HybridRetriever:
    """
    Hybrid retrieval combining dense (ChromaDB) and BM25 searches via RRF.

    Usage:
        retriever = HybridRetriever(chroma_store, bm25_index, embedding_model)
        chunks = retriever.retrieve(query, n_results=20)
        # then pass to reranker...
    """

    def __init__(self, chroma_store, bm25_index, embedding_model, rrf_k: int = 60) -> None:
        """
        Args:
            chroma_store: nexus.indexing.chroma_store.ChromaStore instance.
            bm25_index: nexus.indexing.bm25_index.BM25Index instance.
            embedding_model: nexus.embeddings.model — for query embedding.
            rrf_k: RRF k constant.
        """
        self._store = chroma_store
        self._bm25 = bm25_index
        self._embedder = embedding_model
        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        n_results: int = 20,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-n candidates via hybrid dense+BM25 retrieval with RRF fusion.

        Args:
            query: User query or rewritten query.
            n_results: Number of candidates to return (before reranking).
            where: Optional ChromaDB metadata filter.

        Returns:
            List of RetrievedChunk sorted by RRF score.
        """
        if not query.strip():
            return []

        # 1. Dense retrieval
        query_embedding = self._embedder.embed_query(query)
        chroma_results = self._store.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
        )

        dense_chunks: list[RetrievedChunk] = []
        all_chunks_by_id: dict[str, RetrievedChunk] = {}

        if chroma_results["ids"] and chroma_results["ids"][0]:
            for chunk_id, doc, meta, dist in zip(
                chroma_results["ids"][0],
                chroma_results["documents"][0],
                chroma_results["metadatas"][0],
                chroma_results["distances"][0],
            ):
                chunk = RetrievedChunk.from_chroma_result(chunk_id, doc, meta, dist)
                dense_chunks.append(chunk)
                all_chunks_by_id[chunk_id] = chunk

        # 2. BM25 retrieval
        bm25_results = self._bm25.get_top_n(query, n=n_results)

        # For BM25-only results, fetch their full data from ChromaDB
        bm25_only_ids = {cid for cid, _ in bm25_results if cid not in all_chunks_by_id}
        if bm25_only_ids:
            try:
                fetched = self._store._collection.get(
                    ids=list(bm25_only_ids),
                    include=["documents", "metadatas"],
                )
                for chunk_id, doc, meta in zip(
                    fetched["ids"], fetched["documents"], fetched["metadatas"]
                ):
                    chunk = RetrievedChunk.from_chroma_result(chunk_id, doc, meta, 1.0)
                    all_chunks_by_id[chunk_id] = chunk
            except Exception as e:
                logger.warning("Failed to fetch BM25-only chunks: %s", e)

        # Update BM25 scores in chunk objects
        for chunk_id, score in bm25_results:
            if chunk_id in all_chunks_by_id:
                all_chunks_by_id[chunk_id].bm25_score = score

        # 3. RRF fusion
        fused = reciprocal_rank_fusion(
            dense_results=dense_chunks,
            bm25_results=bm25_results,
            all_chunks_by_id=all_chunks_by_id,
            k=self._rrf_k,
        )

        logger.debug(
            "HybridRetriever: query='%s', dense=%d, bm25=%d, fused=%d",
            query[:60],
            len(dense_chunks),
            len(bm25_results),
            len(fused),
        )

        return fused[:n_results]
