"""
nexus.rerank.cross_encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~
Cross-encoder reranking using sentence-transformers.

What is a cross-encoder and why use it?

  There are two ways to score query-document relevance:

  Bi-encoder (what we use for retrieval):
    - Encode query → vector Q
    - Encode document → vector D
    - Score = cosine_similarity(Q, D)
    - Advantage: pre-compute document vectors; query-time cost = O(1) per doc
    - Disadvantage: Q and D are encoded independently, so the model can't
      capture fine-grained interactions between query tokens and doc tokens

  Cross-encoder (what we use for reranking):
    - Input: "[CLS] query [SEP] document [SEP]" (concatenated)
    - Model: processes both together, attention can flow between Q and D
    - Output: single relevance score
    - Advantage: much more accurate relevance scoring
    - Disadvantage: cannot pre-compute; must run for each (query, doc) pair
    - Cost: O(n) per candidate document

  Strategy: "retrieve-then-rerank"
    1. Dense retrieval: fast, gets top-20 candidates (high recall, lower precision)
    2. Cross-encoder reranking: slow but accurate, reorders top-20 into top-5
    3. Final top-5 go to the LLM

  This gives us near-oracle precision for the LLM context without the
  latency cost of running the cross-encoder on 10,000 documents.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M parameters (same size as bi-encoder MiniLM)
  - Trained on MS MARCO passage retrieval dataset (web search pairs)
  - CPU inference: ~30-50ms per document on modern hardware
  - For 20 candidates: ~0.6-1.0s total reranking latency
  - Strong performance on retrieval benchmarks

Interview answer:
  "We use a two-stage retrieval pipeline. First, a bi-encoder (MiniLM) retrieves
   20 candidates quickly — this maximises recall. Then a cross-encoder reranks
   those 20 into our final top-5. The cross-encoder sees query and document
   together, so it captures semantic interactions that the bi-encoder's
   independent embeddings miss. Our eval showed reranking improved nDCG@5 by
   [X]% at the cost of [Y]ms extra latency, which was acceptable for our use case."
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from nexus.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str = RERANK_MODEL) -> CrossEncoder:
    """
    Load the cross-encoder model (cached — load once per process).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        sentence_transformers.CrossEncoder instance.
    """
    logger.info("Loading cross-encoder: %s", model_name)
    encoder = CrossEncoder(model_name, max_length=512)
    logger.info("Cross-encoder loaded: %s", model_name)
    return encoder


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.

    Takes a list of candidate chunks (from HybridRetriever) and returns
    them re-sorted by cross-encoder relevance score.
    """

    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        """
        Args:
            model_name: HuggingFace cross-encoder model name.
        """
        self.model_name = model_name
        self._encoder = get_cross_encoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Rerank candidate chunks by cross-encoder relevance to the query.

        Args:
            query: The search query (original or rewritten).
            candidates: List of retrieved chunks (from HybridRetriever).
            top_k: Number of top-ranked chunks to return.

        Returns:
            Top-k chunks sorted by cross-encoder score (highest = most relevant).
        """
        if not candidates:
            return []

        if not query.strip():
            return candidates[:top_k]

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, chunk.text) for chunk in candidates]

        # Score all pairs
        # The cross-encoder returns a raw score (not normalised to [0,1])
        # Higher score = more relevant
        scores = self._encoder.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Attach scores to chunks and sort
        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)

        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)

        logger.debug(
            "Reranker: %d candidates → top-%d selected (scores: [%.3f, %.3f])",
            len(candidates),
            top_k,
            ranked[0].rerank_score if ranked else 0,
            ranked[-1].rerank_score if ranked else 0,
        )

        return ranked[:top_k]

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self.model_name})"
