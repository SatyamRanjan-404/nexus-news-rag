"""
tests/unit/test_claims.py
Unit tests for the News Claim Checker models, normalization, and verification logic.
"""

from __future__ import annotations

import pytest

from nexus.claims.models import (
    ClaimCheckRequest,
    ClaimVerdict,
    EvidenceSnippet,
    VerificationResult,
)


@pytest.mark.unit
def test_claim_request_validation():
    req = ClaimCheckRequest(claim="Government announced free 3-month recharge")
    assert req.claim == "Government announced free 3-month recharge"
    assert req.include_evidence is True


@pytest.mark.unit
def test_claim_request_strip_whitespace():
    req = ClaimCheckRequest(claim="   Viral WhatsApp Forward Claim Text   ")
    assert req.claim == "Viral WhatsApp Forward Claim Text"


@pytest.mark.unit
def test_evidence_snippet_model():
    snippet = EvidenceSnippet(
        source_name="BOOM Live",
        title="Debunking 500 Rupee Demonetisation Hoax",
        url="https://boomlive.in/test",
        tier=2,
        snippet_text="RBI confirmed all 500 notes remain legal tender.",
        relevance_score=0.94,
    )
    assert snippet.tier == 2
    assert snippet.relevance_score == 0.94


@pytest.mark.unit
def test_verification_result_serialization():
    res = VerificationResult(
        claim="RBI cancelled 500 rupee notes",
        verdict=ClaimVerdict.FALSE,
        confidence=0.96,
        explanation="RBI has confirmed that no currency cancellation took place.",
        tier2_factcheck_found=True,
        factchecker_reference="BOOM Live rated False",
        tier1_official_confirmed=False,
        independent_sources_count=3,
        evidence=[],
    )
    dumped = res.model_dump()
    assert dumped["verdict"] == "FALSE"
    assert dumped["confidence"] == 0.96
    assert dumped["tier2_factcheck_found"] is True
