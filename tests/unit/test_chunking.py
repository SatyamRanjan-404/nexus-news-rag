"""
tests/unit/test_chunking.py
Unit tests for chunking strategies: Recursive and Semantic chunkers.
"""

from __future__ import annotations

import pytest

from nexus.chunking.recursive import RecursiveChunker
from nexus.ingestion.models import RawArticle


def _create_sample_article(body_text: str) -> RawArticle:
    return RawArticle(
        url="https://example.com/test-story-1",
        title="Test Financial Headline",
        body=body_text,
        extraction_method="full_text",
        source_name="Reuters",
        feed_url="https://reuters.com/feed",
        category="Finance",
        tier=3,
        published_str="2026-10-20T10:00:00",
        published_ts=1792490400,
        url_hash="abc123hash",
    )


@pytest.mark.unit
def test_recursive_chunker_basic_split():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    body = (
        "The Reserve Bank of India decided to maintain the policy repo rate at 6.5 percent today. "
        "The decision was taken by a majority vote of five to one during the Monetary Policy Committee review. "
        "Inflation has moderated towards the target tolerance band of four percent over consecutive months. "
        "Economic output indicators in manufacturing and rural demand continue to show steady expansion."
    )
    article = _create_sample_article(body)
    chunks = chunker.chunk(article)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.chunk_id.startswith("art_abc123hash_chk_")
        assert c.source_url == article.url
        assert c.tier == 3
        assert len(c.text) > 0


@pytest.mark.unit
def test_recursive_chunker_empty_body():
    chunker = RecursiveChunker()
    article = _create_sample_article("")
    chunks = chunker.chunk(article)
    assert chunks == []


@pytest.mark.unit
def test_recursive_chunker_metadata_preservation():
    chunker = RecursiveChunker()
    article = _create_sample_article("Short content statement regarding market trends.")
    chunks = chunker.chunk(article)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["tier"] == 3
    assert chunk.metadata["category"] == "Finance"
    assert chunk.metadata["source"] == "Reuters"
    assert chunk.metadata["published_ts"] == 1792490400
