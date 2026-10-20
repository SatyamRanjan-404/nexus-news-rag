"""
nexus.chunking.semantic
~~~~~~~~~~~~~~~~~~~~~~~
SemanticChunker — splits text at points of high semantic divergence.

How it works (plain English):
  Instead of splitting at a fixed character count, SemanticChunker:
  1. Splits the text into individual sentences (using NLTK/regex)
  2. Embeds each sentence using the same embedding model used for retrieval
  3. Computes cosine similarity between consecutive sentence embeddings
  4. Finds "breakpoints" — places where similarity drops sharply
     (high drop = topic shift = natural chunk boundary)
  5. Groups sentences into chunks based on these breakpoints

Why it can be better than recursive:
  - A chunk always contains sentences about the same sub-topic
  - No mid-sentence or mid-fact splits
  - Better for question answering: the relevant fact is more likely to be
    entirely within one chunk
  - For news: captures "who said what" as a coherent unit

The trade-off:
  - Requires running the embedding model during CHUNKING (slow ingestion)
  - Chunk sizes vary (some chunks may be very short)
  - More expensive on Groq rate limits if average chunk is smaller
  - Requires more careful filtering of tiny chunks

We use LangChain's SemanticChunker as the implementation. It uses a
percentile-based breakpoint threshold: a new chunk starts where the
sentence similarity drop is in the top X% of all drops in the document.

Interview answer:
  "Semantic chunking trades ingestion speed for retrieval quality.
   We ran both strategies on our frozen eval corpus and measured Recall@5.
   Semantic chunking improved recall by Y% on our QA set but doubled
   ingestion time. Given our 15-minute refresh cycle, the extra time
   was acceptable, so we use it in the final pipeline."
   (Fill in Y from actual experiment results.)
"""

from __future__ import annotations

import logging

from langchain_experimental.text_splitter import SemanticChunker

from nexus.chunking.base import BaseChunker
from nexus.ingestion.models import ArticleChunk, RawArticle
from nexus.ingestion.utils import url_to_id

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 80   # drop chunks shorter than this (likely partial sentences)
MIN_ARTICLE_CHARS = 150  # skip articles shorter than this for semantic chunking


class SemanticChunker_(BaseChunker):
    """
    Splits article text at semantic topic boundaries using sentence embeddings.

    This is more expensive than RecursiveChunker but produces more coherent chunks.
    Used in Phase 7 experiments to compare against the recursive baseline.

    The class is named SemanticChunker_ (with underscore) to avoid collision
    with LangChain's SemanticChunker import name.
    """

    def __init__(self, embedding_model, breakpoint_threshold_type: str = "percentile") -> None:
        """
        Args:
            embedding_model: A LangChain Embeddings instance (same model used for retrieval).
                             Passing the same object avoids loading the model twice.
            breakpoint_threshold_type: How to compute the breakpoint threshold.
                "percentile" (default): split where similarity drop is in top 95th percentile
                "standard_deviation": split where drop > 1 std dev from mean
                "interquartile": split at IQR outliers
        """
        self._chunker = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )
        self.breakpoint_threshold_type = breakpoint_threshold_type

    def chunk(self, article: RawArticle) -> list[ArticleChunk]:
        """
        Split a single article at semantic boundaries.

        Args:
            article: RawArticle with a populated body field.

        Returns:
            List of ArticleChunk objects.
        """
        body = article.body.strip()
        if len(body) < MIN_ARTICLE_CHARS:
            logger.debug("Skipping '%s' for semantic chunking (too short)", article.title[:60])
            return []

        try:
            raw_chunks = self._chunker.split_text(body)
        except Exception as exc:
            logger.warning(
                "SemanticChunker failed for '%s': %s — falling back to single chunk",
                article.title[:60],
                exc,
            )
            # Fallback: treat entire article as one chunk
            raw_chunks = [body]

        url_hash = article.url_hash or url_to_id(article.url)

        # Filter out very short chunks (partial sentences from failed splits)
        filtered_chunks = [c for c in raw_chunks if len(c.strip()) >= MIN_CHUNK_CHARS]

        if not filtered_chunks and raw_chunks:
            # If all chunks were filtered, keep the original body as one chunk
            filtered_chunks = [body]

        return [
            ArticleChunk(
                chunk_id=f"{url_hash}_sc{idx}",  # "sc" prefix distinguishes from recursive chunks
                text=chunk_text.strip(),
                chunk_index=idx,
                total_chunks=len(filtered_chunks),
                title=article.title,
                url=article.url,
                source_name=article.source_name,
                category=article.category,
                tier=article.tier,
                published_str=article.published_str,
                published_ts=article.published_ts,
                extraction_method=article.extraction_method,
            )
            for idx, chunk_text in enumerate(filtered_chunks)
        ]

    def __repr__(self) -> str:
        return f"SemanticChunker(breakpoint_type={self.breakpoint_threshold_type})"
