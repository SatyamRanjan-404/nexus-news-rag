"""
nexus.chunking.recursive
~~~~~~~~~~~~~~~~~~~~~~~~
RecursiveCharacterTextSplitter — the original chunking method, now improved.

How RecursiveCharacterTextSplitter works:
  It tries to split text at "natural" boundaries in order of preference:
  1. Double newline (\\n\\n — paragraph boundary)
  2. Single newline (\\n)
  3. Period+space (. — sentence boundary)
  4. Space (word boundary)
  5. Character (last resort — splits mid-word)

  It keeps splitting until all chunks are ≤ chunk_size characters.
  Overlap copies the last `chunk_overlap` characters of chunk N into the
  start of chunk N+1, maintaining context across boundaries.

Why it's good for news:
  News articles have clear paragraph structure. Splitting at paragraphs
  first means each chunk usually contains a coherent sub-story or quote.
  Overlap helps when a key entity is named in the last sentence of one
  paragraph and referred to by pronoun in the next.

The original implementation (rss_fetcher.py) chunked RSS *summaries* —
2-3 sentences of text. Chunking a 100-char summary with chunk_size=500
produces exactly ONE chunk, making chunking pointless. Now that we fetch
full article text, chunking actually splits meaningful content.
"""

from __future__ import annotations

import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter

from nexus.chunking.base import BaseChunker
from nexus.ingestion.models import ArticleChunk, RawArticle
from nexus.ingestion.utils import url_to_id

logger = logging.getLogger(__name__)

MIN_ARTICLE_CHARS = 50  # skip articles shorter than this


class RecursiveChunker(BaseChunker):
    """
    Splits article text using LangChain's RecursiveCharacterTextSplitter.

    This is the BASELINE chunker. All evaluation experiments start from
    this configuration (chunk_size=500, chunk_overlap=50).
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Args:
            chunk_size: Target size in characters for each chunk.
                        Experiments will test 300, 500, 800 chars.
            chunk_overlap: Number of characters to repeat between adjacent chunks.
                           Preserves context across chunk boundaries.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # These are the separators tried in order (most preferred first)
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def chunk(self, article: RawArticle) -> list[ArticleChunk]:
        """
        Split a single article into overlapping text chunks.

        Args:
            article: RawArticle with a populated body field.

        Returns:
            List of ArticleChunk objects.
        """
        body = article.body.strip()
        if len(body) < MIN_ARTICLE_CHARS:
            logger.debug(
                "Skipping article '%s' (body too short: %d chars)",
                article.title[:60],
                len(body),
            )
            return []

        raw_chunks = self._splitter.split_text(body)
        url_hash = article.url_hash or url_to_id(article.url)

        return [
            ArticleChunk(
                chunk_id=f"{url_hash}_c{idx}",
                text=chunk_text,
                chunk_index=idx,
                total_chunks=len(raw_chunks),
                title=article.title,
                url=article.url,
                source_name=article.source_name,
                category=article.category,
                tier=article.tier,
                published_str=article.published_str,
                published_ts=article.published_ts,
                extraction_method=article.extraction_method,
            )
            for idx, chunk_text in enumerate(raw_chunks)
        ]

    def __repr__(self) -> str:
        return f"RecursiveChunker(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
