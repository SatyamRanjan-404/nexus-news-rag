"""
nexus.ingestion.models
~~~~~~~~~~~~~~~~~~~~~~
Data models (dataclasses) for the ingestion pipeline.

Using dataclasses (not Pydantic) here for speed — we process thousands of
articles per cycle and Pydantic validation overhead adds up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class SourceTier(IntEnum):
    """
    Source credibility tier.
    Affects retention policy, citation ordering, and claim checker logic.

    Tier 1: Official/government sources (PIB, RBI, SEBI, ministries)
            Retention: 180 days
    Tier 2: Established fact-checkers (Alt News, BOOM, Factly, PolitiFact, Snopes)
            Retention: 365 days
    Tier 3: Credible general news outlets
            Retention: 14 days
    """

    OFFICIAL = 1
    FACT_CHECKER = 2
    NEWS = 3


class ExtractionMethod(str):
    """How the article body was obtained."""

    FULL_TEXT = "full_text"        # trafilatura successfully extracted full article
    SUMMARY_FALLBACK = "summary_fallback"  # fell back to RSS <summary>


@dataclass
class RawArticle:
    """
    A single article as returned by the RSS fetcher + full-text extractor.
    This is the canonical input to the chunking stage.
    """

    # Identity
    url: str                        # canonical article URL (used as dedup key)
    title: str

    # Content
    body: str                       # full article text (or summary fallback)
    extraction_method: str          # ExtractionMethod constant

    # Provenance
    source_name: str                # human-readable feed name (e.g. "BBC World")
    feed_url: str                   # RSS feed URL
    category: str                   # feed category (World, Business, etc.)
    tier: int                       # SourceTier value

    # Timing
    published_str: str              # ISO-8601 string (e.g. "2026-10-20T14:30:00")
    published_ts: int               # Unix timestamp (int) for fast range filtering in ChromaDB

    # Dedup
    simhash: int = 0                # SimHash fingerprint of body text (Hamming dedup)
    url_hash: str = ""              # MD5(url) — used as the ChromaDB document ID prefix

    # Quality
    word_count: int = 0


@dataclass
class ArticleChunk:
    """
    A single chunk of text from a RawArticle, ready to be embedded and stored.
    Preserves all article metadata so retrieval results are fully cited.
    """

    chunk_id: str                   # globally unique: "{url_hash}_c{chunk_idx}"
    text: str                       # chunk text (500 chars typical)
    chunk_index: int                # position within the article (0-based)
    total_chunks: int               # how many chunks this article was split into

    # Inherited from RawArticle
    title: str
    url: str
    source_name: str
    category: str
    tier: int
    published_str: str
    published_ts: int
    extraction_method: str

    def to_chroma_metadata(self) -> dict:
        """
        Returns a flat dict suitable for ChromaDB metadata storage.
        ChromaDB only accepts str, int, float, bool values — no nested dicts.
        """
        return {
            "title": self.title,
            "link": self.url,
            "source": self.source_name,
            "category": self.category,
            "tier": self.tier,
            "published": self.published_str,
            "published_ts": self.published_ts,
            "extraction_method": self.extraction_method,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


@dataclass
class IngestionStats:
    """Statistics emitted after each ingestion cycle."""

    feed_total: int = 0
    feed_ok: int = 0
    feed_failed: int = 0

    articles_fetched: int = 0
    articles_full_text: int = 0
    articles_summary_fallback: int = 0
    articles_deduped: int = 0
    articles_indexed: int = 0

    chunks_created: int = 0
    chunks_upserted: int = 0

    # Per-tier counts
    tier1_count: int = 0
    tier2_count: int = 0
    tier3_count: int = 0

    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feed_total": self.feed_total,
            "feed_ok": self.feed_ok,
            "feed_failed": self.feed_failed,
            "feed_success_rate": round(self.feed_ok / max(self.feed_total, 1), 3),
            "articles_fetched": self.articles_fetched,
            "articles_full_text": self.articles_full_text,
            "articles_summary_fallback": self.articles_summary_fallback,
            "full_text_rate": round(
                self.articles_full_text / max(self.articles_fetched, 1), 3
            ),
            "articles_deduped": self.articles_deduped,
            "dedup_rate": round(
                self.articles_deduped / max(self.articles_fetched, 1), 3
            ),
            "articles_indexed": self.articles_indexed,
            "chunks_created": self.chunks_created,
            "chunks_upserted": self.chunks_upserted,
            "tier1_count": self.tier1_count,
            "tier2_count": self.tier2_count,
            "tier3_count": self.tier3_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors,
        }
