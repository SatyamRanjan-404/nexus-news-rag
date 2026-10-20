"""
nexus.evaluation.judge
~~~~~~~~~~~~~~~~~~~~~~
LLM-as-a-Judge and heuristic evaluation harness for RAG responses.

Evaluates:
  1. Faithfulness / Groundedness: Are all statements supported by retrieved context?
  2. Answer Relevance: Does the answer address the question?
  3. Citation Correctness: Do the citations support the cited statements?
  4. Abstention Accuracy: Did the model abstain when context was insufficient?

Supports both:
  - Groq LLM-as-a-Judge (production rigorous evaluation)
  - Offline Heuristic Judge (for CI and automated tests without API keys)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvaluationVerdict:
    """Detailed evaluation result for a single generated response."""

    is_grounded: bool
    hallucination_score: float  # 0.0 = fully grounded, 1.0 = fully hallucinated
    unsupported_claims: list[str]
    supported_claims: list[str]
    answer_relevance: float     # 0.0 to 1.0
    citation_precision: float   # 0.0 to 1.0
    abstained_appropriately: bool
    judge_reasoning: str


JUDGE_SYSTEM_PROMPT = """You are an impartial, highly rigorous evaluation judge for a news RAG system.
Your job is to analyze the generated answer against the retrieved context passages and verify if every single factual claim is supported by the context.

Respond ONLY with a valid JSON object with the following schema:
{
  "unsupported_claims": ["list of factual claims in the answer that CANNOT be found in the context"],
  "supported_claims": ["list of claims directly supported by the context"],
  "hallucination_score": <float between 0.0 and 1.0, where 0.0 is completely grounded and 1.0 is completely hallucinated>,
  "answer_relevance": <float between 0.0 and 1.0 representing how directly the answer addresses the question>,
  "citation_precision": <float between 0.0 and 1.0 representing proportion of citations that accurately support their sentence>,
  "abstained_appropriately": <true if the context lacked the answer and the model explicitly stated it cannot answer, otherwise false>,
  "reasoning": "<concise step-by-step reasoning>"
}
"""


class RAGJudge:
    """
    Evaluates groundedness, citations, and relevance of RAG responses.
    """

    def __init__(self, llm_client: Any = None, fallback_mode: bool = False):
        """
        Args:
            llm_client: LangChain chat model (e.g. ChatGroq). If None or error, heuristic mode is used.
            fallback_mode: If True, uses rule-based string overlap heuristic instead of LLM.
        """
        self.llm_client = llm_client
        self.fallback_mode = fallback_mode or (llm_client is None)

    def evaluate(
        self,
        question: str,
        context: str,
        answer: str,
        citations: list[dict] | None = None,
    ) -> EvaluationVerdict:
        """
        Evaluate a RAG response.
        """
        if self.fallback_mode or self.llm_client is None:
            return self._heuristic_evaluate(question, context, answer, citations)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = (
                f"QUESTION: {question}\n\n"
                f"RETRIEVED CONTEXT:\n{context}\n\n"
                f"GENERATED ANSWER:\n{answer}\n\n"
                f"CITATIONS:\n{json.dumps(citations or [], indent=2)}"
            )

            messages = [
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = self.llm_client.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)

            # Extract JSON block
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                hallucination_score = float(data.get("hallucination_score", 0.0))
                return EvaluationVerdict(
                    is_grounded=(hallucination_score < 0.15),
                    hallucination_score=hallucination_score,
                    unsupported_claims=data.get("unsupported_claims", []),
                    supported_claims=data.get("supported_claims", []),
                    answer_relevance=float(data.get("answer_relevance", 1.0)),
                    citation_precision=float(data.get("citation_precision", 1.0)),
                    abstained_appropriately=bool(data.get("abstained_appropriately", False)),
                    judge_reasoning=data.get("reasoning", "LLM-as-a-judge evaluation completed."),
                )
        except Exception as e:
            logger.warning("LLM judge failed (%s); falling back to heuristic evaluation", e)

        return self._heuristic_evaluate(question, context, answer, citations)

    def _heuristic_evaluate(
        self,
        question: str,
        context: str,
        answer: str,
        citations: list[dict] | None = None,
    ) -> EvaluationVerdict:
        """
        Deterministic, fast token-overlap heuristic evaluation for testing.
        """
        abstain_phrases = [
            "cannot find explicit details",
            "not mentioned in today's collected news",
            "insufficient evidence",
            "no reports found",
        ]
        is_abstention = any(p in answer.lower() for p in abstain_phrases)

        if is_abstention:
            # Model abstained
            context_relevance_to_q = any(
                term in context.lower() for term in question.lower().split() if len(term) > 3
            )
            return EvaluationVerdict(
                is_grounded=True,
                hallucination_score=0.0,
                unsupported_claims=[],
                supported_claims=[],
                answer_relevance=1.0 if not context_relevance_to_q else 0.5,
                citation_precision=1.0,
                abstained_appropriately=not context_relevance_to_q,
                judge_reasoning="Model appropriately reported lack of context in corpus.",
            )

        # Sentence-level support check via n-gram overlap
        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 15]
        if not sentences:
            return EvaluationVerdict(
                is_grounded=True,
                hallucination_score=0.0,
                unsupported_claims=[],
                supported_claims=[],
                answer_relevance=0.8,
                citation_precision=1.0,
                abstained_appropriately=False,
                judge_reasoning="Answer contained no evaluable sentences.",
            )

        context_lower = context.lower()
        supported = []
        unsupported = []

        for sent in sentences:
            words = [w.lower() for w in re.findall(r"\b\w{4,}\b", sent)]
            if not words:
                continue
            matched_words = sum(1 for w in words if w in context_lower)
            support_ratio = matched_words / len(words)
            if support_ratio >= 0.5:
                supported.append(sent)
            else:
                unsupported.append(sent)

        total = len(supported) + len(unsupported)
        hallucination_rate = len(unsupported) / total if total > 0 else 0.0

        # Citations check
        has_citations = bool(citations) or bool(re.search(r"\[\d+\]", answer))
        citation_precision = 1.0 if has_citations or not citations else 0.8

        return EvaluationVerdict(
            is_grounded=(hallucination_rate < 0.2),
            hallucination_score=round(hallucination_rate, 4),
            unsupported_claims=unsupported,
            supported_claims=supported,
            answer_relevance=0.9,
            citation_precision=citation_precision,
            abstained_appropriately=False,
            judge_reasoning=f"Heuristic evaluation: {len(supported)}/{total} sentences grounded in context.",
        )
