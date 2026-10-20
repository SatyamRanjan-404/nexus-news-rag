"""
tests/unit/test_ingestion.py
Unit tests for the ingestion layer.

All LLM and network calls are mocked — these tests run fast without API keys.
"""

from __future__ import annotations

import pytest

from nexus.ingestion.dedup import compute_simhash, deduplicate_articles, hamming_distance
from nexus.ingestion.models import RawArticle, SourceTier
from nexus.ingestion.utils import clean_html, normalise_whitespace, url_to_id


# ── HTML cleaning tests ───────────────────────────────────────────

class TestCleanHtml:
    def test_removes_basic_tags(self):
        assert clean_html("<p>Hello world</p>") == "Hello world"

    def test_removes_multiline_tag(self):
        html = '<a href="https://example.com"\nclass="link">Click</a>'
        result = clean_html(html)
        assert "href" not in result
        assert "Click" in result

    def test_decodes_html_entities(self):
        assert ">" in clean_html("&gt;")
        assert "&" in clean_html("&amp;")

    def test_removes_script_content(self):
        html = "<script>alert('xss')</script> Safe text"
        assert "alert" not in clean_html(html)
        assert "Safe text" in clean_html(html)

    def test_empty_string(self):
        assert clean_html("") == ""

    def test_no_html(self):
        assert clean_html("plain text") == "plain text"


# ── URL hashing ───────────────────────────────────────────────────

class TestUrlToId:
    def test_stable(self):
        url = "https://example.com/article/123"
        assert url_to_id(url) == url_to_id(url)

    def test_different_urls(self):
        assert url_to_id("https://a.com") != url_to_id("https://b.com")

    def test_length(self):
        assert len(url_to_id("https://example.com")) == 16


# ── SimHash deduplication ─────────────────────────────────────────

class TestSimHash:
    def test_identical_text_same_hash(self):
        text = "The quick brown fox jumps over the lazy dog"
        assert compute_simhash(text) == compute_simhash(text)

    def test_slightly_different_text_low_hamming(self):
        text1 = "RBI governor announced an interest rate cut of 25 basis points today"
        text2 = "RBI governor announces interest rate cut of 25 basis points today"
        h1 = compute_simhash(text1)
        h2 = compute_simhash(text2)
        dist = hamming_distance(h1, h2)
        assert dist <= 5  # very similar text should have low hamming distance

    def test_different_text_high_hamming(self):
        h1 = compute_simhash("India launches space mission to Mars orbit")
        h2 = compute_simhash("Chef prepares traditional Italian pasta recipe")
        dist = hamming_distance(h1, h2)
        assert dist > 10  # unrelated texts should have high hamming distance


def _make_article(url: str, body: str, tier: int = 3, ts: int = 1000) -> RawArticle:
    return RawArticle(
        url=url,
        title="Test Title",
        body=body,
        extraction_method="full_text",
        source_name="Test Source",
        feed_url="https://test.com/feed",
        category="Test",
        tier=tier,
        published_str="2026-10-20T00:00:00",
        published_ts=ts,
        url_hash=url_to_id(url),
        word_count=len(body.split()),
    )


class TestDeduplicate:
    def test_exact_duplicates_kept_once(self):
        body = "RBI cuts rates by 25 basis points. Markets rally on the announcement."
        a1 = _make_article("https://a.com/1", body, tier=3, ts=1000)
        a2 = _make_article("https://b.com/1", body, tier=3, ts=1001)

        kept, n_dropped = deduplicate_articles([a1, a2])
        assert len(kept) == 1
        assert n_dropped == 1

    def test_keeps_higher_tier(self):
        body = "Breaking news: major earthquake strikes coastal region with 6.2 magnitude."
        a_news = _make_article("https://news.com/1", body, tier=3, ts=1000)
        a_official = _make_article("https://gov.in/1", body, tier=1, ts=1001)

        kept, _ = deduplicate_articles([a_news, a_official])
        assert len(kept) == 1
        assert kept[0].tier == 1  # official source kept over news

    def test_distinct_articles_all_kept(self):
        a1 = _make_article("https://a.com/1", "Apple launches new iPhone with AI features")
        a2 = _make_article("https://b.com/2", "India wins cricket world cup in thrilling final")
        a3 = _make_article("https://c.com/3", "RBI announces monetary policy decision today")

        kept, n_dropped = deduplicate_articles([a1, a2, a3])
        assert len(kept) == 3
        assert n_dropped == 0
