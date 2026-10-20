"""
nexus.chunking.base
~~~~~~~~~~~~~~~~~~~
Abstract base class for all chunking strategies.

Why a common interface?
  The resume claims both "recursive" and "semantic" chunking.
  We implement both, and make them swappable via a config flag.
  The rest of the pipeline only knows about BaseChunker —
  it does not care which strategy is active.
  This lets us run A/B experiments (Phase 7) by just changing one config value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nexus.ingestion.models import ArticleChunk, RawArticle


class BaseChunker(ABC):
    """
    Abstract base class for chunking strategies.

    A chunker takes a RawArticle and returns a list of ArticleChunk objects.
    Each chunk carries all the original article's metadata, so retrieval
    results can be fully cited back to their source.
    """

    @abstractmethod
    def chunk(self, article: RawArticle) -> list[ArticleChunk]:
        """
        Split a single RawArticle into text chunks.

        Args:
            article: A fully-populated RawArticle.

        Returns:
            List of ArticleChunk, possibly empty if article body is too short.
        """
        ...

    def chunk_many(self, articles: list[RawArticle]) -> list[ArticleChunk]:
        """
        Convenience method to chunk a list of articles.
        Delegates to self.chunk() for each article.
        """
        all_chunks: list[ArticleChunk] = []
        for article in articles:
            chunks = self.chunk(article)
            all_chunks.extend(chunks)
        return all_chunks
