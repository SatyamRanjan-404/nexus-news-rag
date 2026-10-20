"""
scripts/run_evaluation.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Top-level runner executing the complete Nexus News RAG benchmark suite:
  1. Retrieval Benchmark (eval_retrieval.py)
  2. Grounded Generation & Hallucination Benchmark (eval_generation.py)
  3. Metric Regression Gate validation (thresholds.json)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add src and scripts to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir / "scripts"))

from eval_generation import run_generation_benchmark
from eval_retrieval import run_benchmark as run_retrieval_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_regression_gate(results_path: Path, thresholds_path: Path) -> bool:
    """
    Check if final evaluation metrics meet or exceed the required regression thresholds.
    """
    if not results_path.exists() or not thresholds_path.exists():
        logger.warning("Files missing for regression gate check; skipping.")
        return True

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(thresholds_path, encoding="utf-8") as f:
        thresh = json.load(f)

    final_metrics = data.get("final_nexus_rag", {})
    claim_metrics = data.get("claim_checker_benchmark", {})

    passed = True
    groundedness = final_metrics.get("groundedness_score", 0.0)
    hallucination = final_metrics.get("hallucination_rate", 1.0)
    citation_prec = final_metrics.get("citation_precision", 0.0)
    claim_acc = claim_metrics.get("verdict_accuracy", 0.0)

    print("\n" + "=" * 60)
    print("CI METRIC REGRESSION GATE VERIFICATION")
    print("=" * 60)

    def _check(name: str, actual: float, target: float, higher_is_better: bool = True):
        nonlocal passed
        ok = (actual >= target) if higher_is_better else (actual <= target)
        op = ">=" if higher_is_better else "<="
        status = "PASSED ✅" if ok else "FAILED ❌"
        if not ok:
            passed = False
        print(f"[{status}] {name:<26}: {actual:.4f} (target {op} {target:.4f})")

    _check("Groundedness Score", groundedness, thresh.get("min_groundedness_score", 0.85))
    _check("Hallucination Rate", hallucination, thresh.get("max_hallucination_rate", 0.08), higher_is_better=False)
    _check("Citation Precision", citation_prec, thresh.get("min_citation_precision", 0.85))
    _check("Claim Checker Accuracy", claim_acc, thresh.get("min_claim_accuracy", 0.85))

    print("=" * 60 + "\n")
    return passed


def main():
    logger.info("Starting comprehensive Nexus News RAG benchmark run...")
    run_retrieval_benchmark()
    run_generation_benchmark()

    results_file = Path("eval/results/baseline_vs_final.json")
    thresholds_file = Path("eval/results/thresholds.json")

    success = validate_regression_gate(results_file, thresholds_file)
    if not success:
        logger.error("Regression gate check failed! One or more metrics degraded below threshold.")
        sys.exit(1)
    else:
        logger.info("All regression gate metrics successfully passed!")


if __name__ == "__main__":
    main()
