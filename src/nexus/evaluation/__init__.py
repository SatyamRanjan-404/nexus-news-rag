"""nexus.evaluation package."""

from nexus.evaluation.judge import EvaluationVerdict, RAGJudge
from nexus.evaluation.metrics import (
    calculate_citation_metrics,
    calculate_classification_metrics,
    calculate_groundedness,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from nexus.evaluation.runner import (
    evaluate_claim_checker_dataset,
    evaluate_generation_dataset,
    evaluate_retrieval_dataset,
)

__all__ = [
    "EvaluationVerdict",
    "RAGJudge",
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "calculate_groundedness",
    "calculate_citation_metrics",
    "calculate_classification_metrics",
    "evaluate_retrieval_dataset",
    "evaluate_generation_dataset",
    "evaluate_claim_checker_dataset",
]
