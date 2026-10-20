"""
nexus.ingestion.dedup
~~~~~~~~~~~~~~~~~~~~~
Two-layer deduplication for ingested articles.

Layer 1 — Exact URL dedup:
    ChromaDB's upsert() is idempotent: reinserting the same chunk_id
    updates the document without creating a duplicate. So exact URL dedup
    is "free" as long as we generate stable chunk IDs from the URL.

Layer 2 — Near-duplicate detection via SimHash:
    The same story is often published by AP, Reuters, PTI, and 10 other
    outlets within minutes with nearly identical text. These all embed to
    nearly the same vector, so retrieval returns 4 near-identical chunks
    that waste context window and reduce diversity.

    SimHash assigns each document a 64-bit integer fingerprint. Documents
    whose fingerprints differ by ≤ 3 bits (Hamming distance) are considered
    near-duplicates. This is O(n) amortised, practical for news corpora.

    When near-duplicates are found, we keep the HIGHEST-TIER source
    (Tier 1 > Tier 2 > Tier 3) since official sources are most credible.
    Within the same tier, we keep the earliest published date.

Why not MinHash LSH?
    MinHash gives a probabilistic set-similarity estimate. It's better for
    very long documents. For news articles (200-2000 words), SimHash is
    simpler and fast enough.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from simhash import Simhash

from nexus.ingestion.models import RawArticle

logger = logging.getLogger(__name__)

# Hamming distance threshold: documents with distance ≤ this are near-duplicates
SIMHASH_THRESHOLD = 3


def compute_simhash(text: str) -> int:
    """
    Compute the 64-bit SimHash fingerprint of a text.

    SimHash works by:
    1. Tokenise the text into n-grams (we use word 2-grams)
    2. Hash each token to a 64-bit value
    3. For each bit position, sum +1 for each token where that bit is 1, -1 otherwise
    4. Final fingerprint: bit=1 if sum>0, else bit=0

    The result is a single integer. Two texts with Hamming distance ≤ 3
    are >95% likely to be near-duplicates for news article length text.

    Args:
        text: Article body text.

    Returns:
        64-bit integer SimHash fingerprint.
    """
    # Use word-level 2-grams for better sensitivity to reordered sentences
    # (common in syndicated rewrites)
    words = text.lower().split()
    features = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)] or words
    return Simhash(features).value


def hamming_distance(a: int, b: int) -> int:
    """
    Count the number of bits that differ between two integers.
    This is the Hamming distance between two SimHash fingerprints.

    Works by XOR-ing the two values (1 where bits differ, 0 where same)
    then counting the set bits (popcount).
    """
    xor = a ^ b
    return bin(xor).count("1")


def deduplicate_articles(articles: list[RawArticle]) -> tuple[list[RawArticle], int]:
    """
    Apply SimHash near-duplicate detection across a list of articles.

    Algorithm:
    1. For each article, compute SimHash of its body text
    2. Compare against all previously seen SimHashes
    3. If Hamming distance ≤ threshold: it's a near-duplicate
       - Group all near-duplicates together
       - Keep only the best representative (highest tier, then earliest date)
    4. Return the deduplicated list and the number of dropped articles

    Args:
        articles: List of RawArticle from fetcher.

    Returns:
        (deduplicated_articles, n_dropped)
    """
    # First, compute SimHash for each article
    for article in articles:
        if article.simhash == 0:  # not yet computed
            article.simhash = compute_simhash(article.body)

    # Group articles by near-duplicate clusters
    # We use a simple O(n²) approach — acceptable for typical ingestion batch size (~2000 articles)
    # For much larger corpora, use an LSH index
    clusters: list[list[RawArticle]] = []
    assigned = [False] * len(articles)

    for i, article in enumerate(articles):
        if assigned[i]:
            continue

        cluster = [article]
        assigned[i] = True

        for j in range(i + 1, len(articles)):
            if assigned[j]:
                continue
            dist = hamming_distance(article.simhash, articles[j].simhash)
            if dist <= SIMHASH_THRESHOLD:
                cluster.append(articles[j])
                assigned[j] = True

        clusters.append(cluster)

    # For each cluster, pick the best representative
    kept: list[RawArticle] = []
    n_dropped = 0

    for cluster in clusters:
        if len(cluster) == 1:
            kept.append(cluster[0])
            continue

        # Sort by: tier ascending (Tier 1 is best), then published_ts ascending (earlier = original)
        best = min(cluster, key=lambda a: (a.tier, a.published_ts))
        kept.append(best)
        dropped = len(cluster) - 1
        n_dropped += dropped

        logger.debug(
            "Dedup cluster (size=%d): kept '%s' [tier=%d], dropped %d near-duplicate(s)",
            len(cluster),
            best.title[:60],
            best.tier,
            dropped,
        )

    logger.info(
        "Deduplication: %d → %d articles (%d dropped as near-duplicates)",
        len(articles),
        len(kept),
        n_dropped,
    )
    return kept, n_dropped


def group_by_story(articles: list[RawArticle]) -> list[list[RawArticle]]:
    """
    Group all articles into story clusters (including duplicates we kept).
    Used by the claim checker to count INDEPENDENT sources:
    articles within the same SimHash cluster count as ONE source.

    Returns:
        List of clusters, each cluster is a list of RawArticle covering the same story.
    """
    # Compute simhash if not done yet
    for article in articles:
        if article.simhash == 0:
            article.simhash = compute_simhash(article.body)

    clusters: list[list[RawArticle]] = []
    assigned = [False] * len(articles)

    for i, article in enumerate(articles):
        if assigned[i]:
            continue
        cluster = [article]
        assigned[i] = True
        for j in range(i + 1, len(articles)):
            if assigned[j]:
                continue
            if hamming_distance(article.simhash, articles[j].simhash) <= SIMHASH_THRESHOLD:
                cluster.append(articles[j])
                assigned[j] = True
        clusters.append(cluster)

    return clusters
