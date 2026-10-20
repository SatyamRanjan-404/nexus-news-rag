"""
nexus.indexing.bm25_index
~~~~~~~~~~~~~~~~~~~~~~~~~~
BM25 keyword retrieval index for hybrid search.

What is BM25?
  BM25 (Best Match 25) is a probabilistic ranking function used by search engines
  like Elasticsearch and Lucene. It ranks documents by term frequency and inverse
  document frequency (TF-IDF), adjusted for document length.

  Formula (simplified):
    score(D, Q) = Σ IDF(qi) * (tf(qi, D) * (k1 + 1)) / (tf(qi, D) + k1 * (1 - b + b * |D|/avgdl))

  Where:
    - tf(qi, D): frequency of term qi in document D
    - IDF(qi): inverse document frequency (rare terms score higher)
    - |D|: document length in tokens
    - avgdl: average document length
    - k1=1.5, b=0.75: tuning parameters (BM25+ defaults)

Why BM25 for news?
  Dense embeddings (MiniLM) capture semantic similarity well but can miss
  exact entity matches. For example, searching "₹2.5 lakh crore budget"
  — the embedding may find articles about "government spending" while
  missing the article that specifically mentions that exact figure.
  BM25 finds exact matches like entity names, numbers, and dates.

  Hybrid retrieval (BM25 + dense) via RRF gets the best of both:
  semantic understanding AND exact matching.

Implementation:
  We use rank_bm25 (pure Python, no Java/Elasticsearch needed).
  The index is rebuilt from the ChromaDB corpus at startup (~1s for 10k docs).
  This is acceptable because the index is tiny in memory and fast to rebuild.

Why rebuild at startup instead of persisting?
  rank_bm25 doesn't have a built-in serialisation method. We could pickle it,
  but the ChromaDB corpus is always fresher. Rebuilding ensures the BM25 index
  and the vector index are always in sync.
"""

from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def simple_tokenize(text: str) -> list[str]:
    """
    Basic word tokeniser: lowercase, split on non-alphanumeric characters.
    Simple and fast — adequate for BM25 (which is a bag-of-words model).
    """
    return re.findall(r"[a-z0-9₹$€£¥%]+", text.lower())


class BM25Index:
    """
    In-memory BM25 retrieval index built from a corpus of text documents.

    Lifecycle:
      1. On ingestion pipeline startup: call BM25Index.build_from_chroma(store)
      2. During retrieval: call index.get_top_n(query, n) → ranked doc IDs
      3. RRF fusion combines BM25 ranks with dense retrieval ranks

    The index stores document IDs alongside the BM25 scores so we can
    match results back to ChromaDB records.
    """

    def __init__(self, corpus: list[str], doc_ids: list[str]) -> None:
        """
        Build the BM25 index from a corpus.

        Args:
            corpus: List of document text strings.
            doc_ids: Corresponding list of document IDs (must be same length).
        """
        assert len(corpus) == len(doc_ids), "corpus and doc_ids must have same length"

        self.doc_ids = doc_ids
        self._tokenized_corpus = [simple_tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info("BM25 index built: %d documents", len(corpus))

    def get_top_n(self, query: str, n: int = 20) -> list[tuple[str, float]]:
        """
        Return the top-n documents by BM25 score for a query.

        Args:
            query: Query string.
            n: Number of results to return.

        Returns:
            List of (doc_id, bm25_score) tuples, sorted by score descending.
            Scores are raw BM25 values (not normalised).
        """
        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Sort by score descending, return top-n
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for idx, score in ranked[:n]:
            if score > 0:  # BM25 score of 0 = no term overlap = irrelevant
                results.append((self.doc_ids[idx], float(score)))

        return results

    @classmethod
    def build_from_chroma(cls, store) -> "BM25Index":
        """
        Build a BM25Index from all documents currently in a ChromaStore.

        Args:
            store: ChromaStore instance.

        Returns:
            Populated BM25Index.
        """
        data = store.get_all_documents()
        doc_ids = data.get("ids", [])
        documents = data.get("documents", [])

        if not documents:
            logger.warning("ChromaDB is empty — BM25 index will be empty")
            return cls([], [])

        return cls(corpus=documents, doc_ids=doc_ids)

    def __len__(self) -> int:
        return len(self.doc_ids)
