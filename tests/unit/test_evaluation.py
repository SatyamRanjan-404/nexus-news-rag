"""
tests/unit/test_evaluation.py
Unit tests for evaluation metrics: Recall@K, Precision@K, MRR@K, NDCG@K,
groundedness scoring, and citation metrics.
"""

from __future__ import annotations

import pytest

from nexus.evaluation.metrics import (
    calculate_citation_metrics,
    calculate_classification_metrics,
    calculate_groundedness,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@pytest.mark.unit
def test_recall_at_k():
    retrieved = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
    gold = {"doc_B", "doc_D", "doc_Z"}

    # Top 3 has doc_B (1 of 3)
    assert recall_at_k(retrieved, gold, k=3) == pytest.approx(1 / 3)
    # Top 5 has doc_B, doc_D (2 of 3)
    assert recall_at_k(retrieved, gold, k=5) == pytest.approx(2 / 3)


@pytest.mark.unit
def test_precision_at_k():
    retrieved = ["doc_A", "doc_B", "doc_C", "doc_D"]
    gold = {"doc_B", "doc_D"}

    # Top 2 has doc_B (1 of 2)
    assert precision_at_k(retrieved, gold, k=2) == pytest.approx(0.5)
    # Top 4 has doc_B, doc_D (2 of 4)
    assert precision_at_k(retrieved, gold, k=4) == pytest.approx(0.5)


@pytest.mark.unit
def test_mrr_at_k():
    retrieved = ["doc_A", "doc_B", "doc_C"]
    gold = {"doc_B"}

    # First relevant chunk is at rank 2 -> MRR = 1/2 = 0.5
    assert mrr_at_k(retrieved, gold, k=3) == pytest.approx(0.5)

    # First relevant chunk is at rank 1 -> MRR = 1.0
    assert mrr_at_k(["doc_B", "doc_A"], gold, k=3) == pytest.approx(1.0)

    # No relevant chunk -> MRR = 0.0
    assert mrr_at_k(["doc_X", "doc_Y"], gold, k=3) == pytest.approx(0.0)


@pytest.mark.unit
def test_ndcg_at_k():
    # Ideal ranking: relevant at rank 1
    retrieved = ["doc_A", "doc_B"]
    gold = {"doc_A"}
    assert ndcg_at_k(retrieved, gold, k=2) == pytest.approx(1.0)


@pytest.mark.unit
def test_calculate_groundedness():
    # 2 unsupported out of 10 sentences -> 80% groundedness, 20% hallucination
    groundedness, hallucination = calculate_groundedness(unsupported_sentences=2, total_sentences=10)
    assert groundedness == pytest.approx(0.80)
    assert hallucination == pytest.approx(0.20)


@pytest.mark.unit
def test_calculate_citation_metrics():
    metrics = calculate_citation_metrics(
        valid_citations=9,
        total_citations=10,
        supported_claims_with_citations=8,
        total_supported_claims=10,
    )
    assert metrics["citation_precision"] == 0.9
    assert metrics["citation_recall"] == 0.8
    assert metrics["citation_f1"] > 0.8


@pytest.mark.unit
def test_calculate_classification_metrics():
    y_true = ["TRUE", "FALSE", "FALSE", "UNVERIFIED"]
    y_pred = ["TRUE", "FALSE", "TRUE", "UNVERIFIED"]  # 3 of 4 correct
    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert metrics["sample_size"] == 4
