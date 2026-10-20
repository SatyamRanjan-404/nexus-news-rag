"""
nexus.ingestion.utils
~~~~~~~~~~~~~~~~~~~~~
Shared utilities for the ingestion pipeline.

- HTML cleaning via BeautifulSoup (not regex — see ADR-004)
- URL normalisation
- Text normalisation
- Date parsing helpers
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone

from bs4 import BeautifulSoup


def clean_html(raw_html: str) -> str:
    """
    Remove HTML tags and decode HTML entities from a string.

    Why BeautifulSoup instead of regex?
    regex like `<.*?>` fails on:
      - Multi-line tags: <a href="..."\\nclass="...">
      - HTML entities: &amp; &nbsp; &lt; are NOT removed by tag-stripping regex
      - Script/style blocks: their text content is left behind
    BeautifulSoup handles all of these correctly.

    Args:
        raw_html: String that may contain HTML markup.

    Returns:
        Plain text with tags removed, entities decoded, and whitespace normalised.
    """
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    # Remove script and style elements entirely (including their text)
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return normalise_whitespace(text)


def normalise_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/newlines into a single space and strip edges.
    Also normalises Unicode whitespace (non-breaking spaces, etc.).
    """
    # Normalise unicode
    text = unicodedata.normalize("NFKC", text)
    # Replace all whitespace sequences with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def url_to_id(url: str) -> str:
    """
    Returns a stable 16-char hex string derived from the URL.
    Used as a prefix for chunk IDs to enable idempotent upserts.
    Two chunks from the same article will share the same prefix.

    Args:
        url: Canonical article URL.

    Returns:
        16-character hex string (first 8 bytes of MD5).
    """
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:16]


def parse_published_date(time_struct) -> tuple[str, int]:
    """
    Convert a feedparser time_struct to (ISO string, Unix timestamp).

    feedparser returns a time.struct_time in UTC for the published_parsed field.
    We use calendar.timegm() (not time.mktime()) to correctly interpret UTC.

    Args:
        time_struct: time.struct_time from feedparser entry.published_parsed,
                     or None if the field is missing.

    Returns:
        (iso_string, unix_timestamp) — falls back to current UTC time if None.
    """
    import calendar

    if time_struct is not None:
        try:
            dt = datetime(*time_struct[:6], tzinfo=timezone.utc)
            return dt.isoformat(), calendar.timegm(time_struct)
        except (ValueError, TypeError):
            pass

    # Fallback: use current UTC time
    now = datetime.now(tz=timezone.utc)
    return now.isoformat(), int(now.timestamp())


def estimate_word_count(text: str) -> int:
    """Fast word count estimate."""
    return len(text.split())
