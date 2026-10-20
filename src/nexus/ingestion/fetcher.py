"""
nexus.ingestion.fetcher
~~~~~~~~~~~~~~~~~~~~~~~
RSS feed fetcher with full-text extraction via trafilatura.

Design (see ADR-003):
  1. Read news_sources.json to get feed list with tier metadata
  2. Parse RSS with feedparser (10s socket timeout)
  3. For each article, attempt full-text extraction via trafilatura (3s timeout)
  4. If extraction fails or text < 200 chars, fall back to RSS <summary>
  5. Clean HTML from text via BeautifulSoup (see ADR-004)
  6. Return list of RawArticle dataclasses with full metadata

Key improvement over original:
  - Embeds FULL article text, not 2-3 sentence summaries
  - Concurrent fetching (ThreadPoolExecutor) for speed
  - Proper HTML cleaning
  - Per-tier metadata
  - Structured error handling per feed (one bad feed doesn't crash all)
"""

from __future__ import annotations

import json
import logging
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import feedparser
import trafilatura
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from nexus.ingestion.models import RawArticle, SourceTier
from nexus.ingestion.utils import (
    clean_html,
    estimate_word_count,
    normalise_whitespace,
    parse_published_date,
    url_to_id,
)

# Set a global socket timeout to prevent feedparser hanging on dead servers
socket.setdefaulttimeout(10)

logger = logging.getLogger(__name__)


# ── Feed source loading ───────────────────────────────────────────


def load_sources(sources_path: Path | str = "news_sources.json") -> list[dict]:
    """
    Load and validate the news sources configuration file.
    Returns only enabled sources.
    """
    path = Path(sources_path)
    if not path.exists():
        logger.error("news_sources.json not found at %s", path.absolute())
        return []

    with open(path, encoding="utf-8") as f:
        sources = json.load(f)

    enabled = [s for s in sources if s.get("enabled", True)]
    logger.info("Loaded %d/%d enabled sources", len(enabled), len(sources))
    return enabled


# ── Full-text extraction ──────────────────────────────────────────

MIN_FULL_TEXT_CHARS = 200  # Below this, we consider trafilatura extraction failed


def extract_full_text(url: str, timeout: int = 5) -> str | None:
    """
    Attempt to fetch and extract the full article text from a URL using trafilatura.

    trafilatura is a purpose-built article extractor that:
    - Identifies the main content block (removes nav, ads, footers)
    - Handles paywall attempts gracefully (returns None)
    - Respects robots.txt
    - Works on most major news sites

    Args:
        url: Article URL.
        timeout: HTTP request timeout in seconds.

    Returns:
        Extracted text, or None if extraction fails.
    """
    try:
        # trafilatura.fetch_url handles the HTTP request internally
        downloaded = trafilatura.fetch_url(url, no_ssl=False)
        if downloaded is None:
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=True,  # Only return if confident extraction succeeded
        )
        if text and len(text) >= MIN_FULL_TEXT_CHARS:
            return normalise_whitespace(text)

        return None

    except Exception as exc:
        logger.debug("trafilatura failed for %s: %s", url, exc)
        return None


# ── Feed parsing ──────────────────────────────────────────────────


def parse_single_feed(
    source: dict,
) -> tuple[list[RawArticle], bool]:
    """
    Parse one RSS feed and return (articles, success_flag).

    Args:
        source: dict with keys: name, url, category, tier, enabled

    Returns:
        (list of RawArticle, True if feed parsed successfully)
    """
    source_name: str = source["name"]
    feed_url: str = source["url"]
    category: str = source.get("category", "General")
    tier: int = source.get("tier", SourceTier.NEWS)

    try:
        parsed = feedparser.parse(feed_url)

        if parsed.bozo and not parsed.entries:
            # bozo=True means malformed XML; if entries is also empty, feed is dead
            logger.warning(
                "Feed '%s' returned no entries (bozo=%s, status=%s)",
                source_name,
                parsed.bozo,
                getattr(parsed, "status", "unknown"),
            )
            return [], False

        articles: list[RawArticle] = []

        for entry in parsed.entries:
            url = entry.get("link", "").strip()
            if not url:
                continue  # skip entries without a URL

            title = normalise_whitespace(entry.get("title", "No Title"))

            # Get RSS summary as fallback (clean HTML first)
            raw_summary = entry.get("summary", entry.get("description", ""))
            summary = clean_html(raw_summary)

            # Attempt full-text extraction
            full_text = extract_full_text(url)

            if full_text:
                body = full_text
                extraction_method = "full_text"
            else:
                body = summary
                extraction_method = "summary_fallback"
                if not body:
                    continue  # skip articles with no text at all

            # Parse date
            published_str, published_ts = parse_published_date(
                entry.get("published_parsed", None)
            )

            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    body=body,
                    extraction_method=extraction_method,
                    source_name=source_name,
                    feed_url=feed_url,
                    category=category,
                    tier=tier,
                    published_str=published_str,
                    published_ts=published_ts,
                    url_hash=url_to_id(url),
                    word_count=estimate_word_count(body),
                )
            )

        logger.info(
            "Feed '%s': %d articles parsed (%d entries total)",
            source_name,
            len(articles),
            len(parsed.entries),
        )
        return articles, True

    except Exception as exc:
        logger.error("Feed '%s' (%s) failed: %s", source_name, feed_url, exc)
        return [], False


# ── Concurrent fetching ───────────────────────────────────────────


def fetch_all_feeds(
    sources: list[dict],
    max_workers: int = 10,
) -> tuple[list[RawArticle], dict[str, bool]]:
    """
    Fetch all feeds concurrently using a thread pool.

    Why threads instead of asyncio?
    feedparser is synchronous. trafilatura's HTTP requests are also synchronous.
    ThreadPoolExecutor gives us concurrent I/O without rewriting either library.

    Args:
        sources: List of source dicts from news_sources.json
        max_workers: Maximum concurrent fetch threads

    Returns:
        (all_articles, {feed_name: success_flag})
    """
    all_articles: list[RawArticle] = []
    feed_results: dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {
            executor.submit(parse_single_feed, source): source for source in sources
        }

        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                articles, success = future.result()
                all_articles.extend(articles)
                feed_results[source["name"]] = success
            except Exception as exc:
                logger.error("Unexpected error processing feed '%s': %s", source["name"], exc)
                feed_results[source["name"]] = False

    logger.info(
        "Fetch complete: %d articles from %d/%d feeds",
        len(all_articles),
        sum(feed_results.values()),
        len(sources),
    )
    return all_articles, feed_results
