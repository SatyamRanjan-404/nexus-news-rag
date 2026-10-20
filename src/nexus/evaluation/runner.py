"""
nexus.evaluation.runner
~~~~~~~~~~~~~~~~~~~~~~~
Top-level evaluation suite runner: executes retrieval, generation, and claim
verification benchmark pipelines and computes aggregate metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nexus.evaluation.judge import RAGJudge
from nexus.evaluation.metrics import (
    calculate_classification_metrics,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


def evaluate_retrieval_dataset(
    dataset_path: str | Path,
    retrieval_func: Any,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """
    Run retrieval evaluation across test queries in dataset_path.

    Args:
        dataset_path: JSON file containing list of test cases with fields:
                      {"id": "...", "query": "...", "relevant_chunk_ids": ["..."]}
        retrieval_func: Callable accepting (query, top_k) -> list[RetrievedChunk] or list[str chunk_ids]
        k_values: Tuple of K cutoffs to evaluate.

    Returns:
        Aggregated Recall@K, Precision@K, MRR@K, NDCG@K metrics.
    """
    path = Path(dataset_path)
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)

    results: dict[str, list[float]] = {}
    for k in k_values:
        results[f"recall@{k}"] = []
        results[f"precision@{k}"] = []
        results[f"mrr@{k}"] = []
        results[f"ndcg@{k}"] = []

    for case in cases:
        query = case["query"]
        gold_ids = set(case.get("relevant_chunk_ids", []))
        if not gold_ids:
            continue

        raw_retrieved = retrieval_func(query, top_k=max(k_values))
        retrieved_ids = [
            getattr(item, "chunk_id", str(item)) for item in raw_retrieved
        ]

        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(retrieved_ids, gold_ids, k=k))
            results[f"precision@{k}"].append(precision_at_k(retrieved_ids, gold_ids, k=k))
            results[f"mrr@{k}"].append(mrr_at_k(retrieved_ids, gold_ids, k=k))
            results[f"ndcg@{k}"].append(ndcg_at_k(retrieved_ids, gold_ids, k=k))

    summary = {
        "num_test_cases": len(cases),
        "metrics": {
            metric: round(sum(vals) / len(vals), 4) if vals else 0.0
            for metric, vals in results.items()
        },
    }
    return summary


def evaluate_generation_dataset(
    dataset_path: str | Path,
    rag_chain_func: Any,
    judge: RAGJudge,
) -> dict[str, Any]:
    """
    Run groundedness and hallucination evaluation across QA test cases.

    Args:
        dataset_path: JSON file containing QA test cases.
        rag_chain_func: Callable accepting (query) -> {"answer": str, "context": str, "citations": list}
        judge: RAGJudge instance.

    Returns:
        Aggregated groundedness, hallucination rate, and citation precision.
    """
    path = Path(dataset_path)
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)

    hallucination_scores = []
    grounded_count = 0
    citation_precisions = []
    verdicts = []

    for case in cases:
        query = case["question"]
        output = rag_chain_func(query)

        answer = output.get("answer", "")
        context = output.get("context", "")
        citations = output.get("citations", [])

        verdict = judge.evaluate(
            question=query,
            context=context,
            answer=answer,
            citations=citations,
        )

        hallucination_scores.append(verdict.hallucination_score)
        if verdict.is_grounded:
            grounded_count += 1
        citation_precisions.append(verdict.citation_precision)
        verdicts.append(asdict(verdict))

    n = len(cases)
    avg_hallucination = sum(hallucination_scores) / n if n else 0.0
    groundedness_ratio = grounded_count / n if n else 0.0
    avg_citation_precision = sum(citation_precisions) / n if n else 0.0

    return {
        "num_test_cases": n,
        "groundedness_rate": round(groundedness_ratio, 4),
        "hallucination_rate": round(avg_hallucination, 4),
        "citation_precision": round(avg_citation_precision, 4),
        "detailed_verdicts": verdicts,
    }


def evaluate_claim_checker_dataset(
    dataset_path: str | Path,
    claim_checker_func: Any,
) -> dict[str, Any]:
    """
    Evaluate Claim Checker on gold-annotated news rumors and forwards.
    Ensures held-out fact-checking sources are isolated to prevent leakage.
    """
    path = Path(dataset_path)
    with open(path, encoding="utf-8") as f:
        claims = json.load(f)

    y_true = []
    y_pred = []
    detailed = []

    for item in claims:
        claim_text = item["claim"]
        gold_verdict = item["gold_verdict"]

        # Run verification pipeline
        result = claim_checker_func(claim_text)
        predicted_verdict = getattr(result, "verdict", str(result))
        if hasattr(predicted_verdict, "value"):
            predicted_verdict = predicted_verdict.value

        y_true.append(gold_verdict)
        y_pred.append(predicted_verdict)

        detailed.append({
            "claim": claim_text,
            "gold_verdict": gold_verdict,
            "predicted_verdict": predicted_verdict,
            "is_correct": (gold_verdict == predicted_verdict),
            "held_out_source": item.get("held_out_source", None),
        })

    metrics = calculate_classification_metrics(y_true, y_pred)
    return {
        "num_claims": len(claims),
        "overall_metrics": metrics,
        "results": detailed,
    }
