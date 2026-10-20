"""
nexus.evaluation.metrics
~~~~~~~~~~~~~~~~~~~~~~~~
Evaluation metrics for Retrieval quality, RAG groundedness, and Claim Verification.

Metrics implemented:
1. Retrieval Metrics:
   - Recall@K: Proportion of relevant chunks retrieved in top K.
   - MRR@K (Mean Reciprocal Rank): 1 / rank of first relevant chunk.
   - NDCG@K (Normalized Discounted Cumulative Gain): Positional relevance weighting.
   - Precision@K: Proportion of retrieved chunks in top K that are relevant.

2. Groundedness & Generation Metrics:
   - Hallucination Rate: Fraction of generated sentences not supported by retrieved context.
   - Groundedness Score: (1 - Hallucination Rate).
   - Citation Precision: Fraction of citations that genuinely support their assertion.
   - Citation Recall: Fraction of supported claims that carry an explicit citation.

3. Claim Verification Metrics:
   - Accuracy, Precision, Recall, Macro-F1 across verdicts (TRUE, FALSE, MISLEADING, UNVERIFIED).
   - Abstention Rate: Proportion of claims where the model correctly abstains due to low evidence.
"""

from __future__ import annotations

import math
from typing import Sequence


# ── Retrieval Metrics ────────────────────────────────────────────────


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str] | set[str],
    k: int = 5,
) -> float:
    """
    Calculate Recall@K:
    Proportion of ground-truth relevant chunk IDs that appear in top-K retrieved IDs.
    """
    if not relevant_ids:
        return 1.0
    relevant_set = set(relevant_ids)
    top_k = set(retrieved_ids[:k])
    hits = len(top_k.intersection(relevant_set))
    return float(hits / len(relevant_set))


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str] | set[str],
    k: int = 5,
) -> float:
    """
    Calculate Precision@K:
    Proportion of top-K retrieved chunks that are relevant.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return float(hits / len(top_k))


def mrr_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str] | set[str],
    k: int = 5,
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR@K):
    1 / position (1-indexed) of the first relevant chunk in top-K.
    Returns 0.0 if no relevant chunk is found within top-K.
    """
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str] | set[str],
    k: int = 5,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K) under binary relevance.
    """
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    dcg = 0.0
    for i, cid in enumerate(top_k):
        rel = 1.0 if cid in relevant_set else 0.0
        dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because i is 0-indexed

    # Ideal DCG: all relevant items ranked first
    idcg = 0.0
    num_ideal = min(len(relevant_set), k)
    for i in range(num_ideal):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


# ── Groundedness & Hallucination Metrics ─────────────────────────────


def calculate_groundedness(
    unsupported_sentences: int,
    total_sentences: int,
) -> tuple[float, float]:
    """
    Returns (groundedness_score, hallucination_rate).
    Groundedness score is in [0.0, 1.0].
    Hallucination rate is the fraction of sentences unsupported by retrieved context.
    """
    if total_sentences <= 0:
        return 1.0, 0.0

    hallucination_rate = min(1.0, max(0.0, unsupported_sentences / total_sentences))
    groundedness_score = 1.0 - hallucination_rate
    return float(groundedness_score), float(hallucination_rate)


def calculate_citation_metrics(
    valid_citations: int,
    total_citations: int,
    supported_claims_with_citations: int,
    total_supported_claims: int,
) -> dict[str, float]:
    """
    Citation Precision: valid_citations / total_citations
    Citation Recall: supported_claims_with_citations / total_supported_claims
    """
    precision = (valid_citations / total_citations) if total_citations > 0 else 1.0
    recall = (
        (supported_claims_with_citations / total_supported_claims)
        if total_supported_claims > 0
        else 1.0
    )
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "citation_precision": round(float(precision), 4),
        "citation_recall": round(float(recall), 4),
        "citation_f1": round(float(f1), 4),
    }


# ── Classification / Verification Metrics ────────────────────────────


def calculate_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = ("TRUE", "FALSE", "MISLEADING", "UNVERIFIED"),
) -> dict[str, float]:
    """
    Calculate Accuracy, per-class Precision/Recall, and Macro-F1 across verdicts.
    """
    if not y_true or len(y_true) != len(y_pred):
        return {"accuracy": 0.0, "macro_f1": 0.0}

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total

    f1_scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "sample_size": total,
    }
