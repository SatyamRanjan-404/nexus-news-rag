"""
tests/unit/test_retrieval.py
Unit tests for the retrieval layer.
"""

from __future__ import annotations

import pytest

from nexus.retrieval.hybrid import RetrievedChunk, reciprocal_rank_fusion


def _make_chunk(chunk_id: str, dense_score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"Text for chunk {chunk_id}",
        title="Test Article",
        url=f"https://example.com/{chunk_id}",
        source_name="Test",
        category="General",
        tier=3,
        published_str="2026-10-20T00:00:00",
        published_ts=1000000,
        extraction_method="full_text",
        dense_score=dense_score,
    )


class TestRRF:
    def test_both_lists_bonus(self):
        """A doc in both lists should score higher than a doc in only one."""
        chunk_a = _make_chunk("a", dense_score=0.9)
        chunk_b = _make_chunk("b", dense_score=0.7)

        # a is rank 1 in dense, rank 1 in BM25 → gets double RRF bonus
        # b is rank 2 in dense, not in BM25
        dense_results = [chunk_a, chunk_b]
        bm25_results = [("a", 10.0), ("c", 8.0)]

        chunk_c = _make_chunk("c")
        all_by_id = {"a": chunk_a, "b": chunk_b, "c": chunk_c}

        fused = reciprocal_rank_fusion(dense_results, bm25_results, all_by_id, k=60)

        # a should be first (in both lists)
        assert fused[0].chunk_id == "a"

    def test_empty_lists(self):
        result = reciprocal_rank_fusion([], [], {}, k=60)
        assert result == []

    def test_dense_only(self):
        chunk_a = _make_chunk("a")
        chunk_b = _make_chunk("b")
        fused = reciprocal_rank_fusion([chunk_a, chunk_b], [], {"a": chunk_a, "b": chunk_b}, k=60)
        assert len(fused) == 2
        assert fused[0].chunk_id == "a"  # rank 1 in dense

    def test_rrf_scores_decrease_with_rank(self):
        """Lower rank → lower RRF contribution."""
        # 1/(60+1) > 1/(60+2) > 1/(60+3)
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        all_by_id = {c.chunk_id: c for c in chunks}
        fused = reciprocal_rank_fusion(chunks, [], all_by_id, k=60)
        scores = [c.rrf_score for c in fused]
        assert scores[0] > scores[1] > scores[2]
