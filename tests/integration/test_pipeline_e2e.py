"""
tests/integration/test_pipeline_e2e.py
End-to-end integration test for the Nexus RAG pipeline with mock components.
"""

from __future__ import annotations

import pytest

from nexus.chunking.recursive import RecursiveChunker
from nexus.evaluation.judge import RAGJudge
from nexus.ingestion.models import RawArticle


@pytest.mark.integration
def test_e2e_chunk_and_evaluate_flow():
    # 1. Ingestion article
    article = RawArticle(
        url="https://rbi.org.in/press/mpc-oct.html",
        title="RBI Monetary Policy Statement",
        body="The Monetary Policy Committee of the Reserve Bank of India kept the repo rate unchanged at 6.50 percent.",
        extraction_method="full_text",
        source_name="RBI Press Releases",
        feed_url="https://rbi.org.in/feed.xml",
        category="Finance",
        tier=1,
        published_str="2026-10-09T11:00:00",
        published_ts=1791543600,
        url_hash="rbi_test_hash",
    )

    # 2. Chunking
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk(article)
    assert len(chunks) >= 1

    # 3. Generation evaluation
    judge = RAGJudge(fallback_mode=True)
    verdict = judge.evaluate(
        question="What is the repo rate?",
        context=article.body,
        answer="The Reserve Bank of India maintained the repo rate at 6.50 percent [1].",
        citations=[{"title": article.title, "link": article.url}],
    )

    assert verdict.is_grounded is True
    assert verdict.hallucination_score == 0.0
