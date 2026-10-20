"""
nexus.claims.models
~~~~~~~~~~~~~~~~~~~~
Pydantic models for the claim checker.

Using Pydantic here (not dataclasses) because:
1. These are API response models — they must be JSON-serialisable
2. Pydantic V2's model_json_schema() is used for structured LLM output
3. FastAPI uses Pydantic models for request/response validation

Design:
  The claim checker returns a ClaimVerdict with:
  - verdict: one of SUPPORTED / CONTRADICTED / NOT_ENOUGH_EVIDENCE
  - confidence: 0.0-1.0 float
  - explanation: human-readable summary in 2-3 sentences
  - evidence: list of EvidenceSnippet (cited quotes from articles)
  - independent_sources: count of INDEPENDENT sources (wire copy deduped)
  - freshness_note: comment on how recent the evidence is
  - atomic_claims: breakdown of the input into checkable sub-claims

Claim checker honesty:
  The verdict refers to what the RETRIEVED ARTICLES say, not absolute truth.
  "SUPPORTED" means "the retrieved articles support this claim."
  This distinction is explicitly stated in the UI, API, and README.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Verdict(str, Enum):
    """
    Three possible outcomes for a claim check.

    SUPPORTED: The retrieved evidence supports the claim.
               ≥ 1 independent source confirms the claim's key facts.

    CONTRADICTED: The retrieved evidence contradicts the claim.
                  ≥ 1 independent source states something materially different
                  from what the claim asserts.

    NOT_ENOUGH_EVIDENCE: The retrieved articles do not contain sufficient
                         information to support or contradict the claim.
                         This is returned when:
                         - No relevant articles were found
                         - Reranker confidence is below threshold
                         - Evidence is too old (outside retention window)
                         - The claim is about a future event or opinion
    """

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_EVIDENCE = "NOT_ENOUGH_EVIDENCE"


class EvidenceSnippet(BaseModel):
    """
    A single piece of cited evidence from a retrieved article.

    The quote field must be a verbatim excerpt from the source article.
    After the LLM returns a verdict, we verify each quote is actually
    present in the retrieved text (programmatic citation verification).
    Quotes that fail verification are removed.
    """

    source_name: str = Field(description="Feed/outlet name (e.g. 'BBC World')")
    title: str = Field(description="Article headline")
    url: str = Field(description="Article URL")
    published_date: str = Field(description="Publication date (YYYY-MM-DD)")
    quote: str = Field(
        description="Verbatim excerpt from the article that supports or contradicts the claim. "
        "Must be an exact substring of the article text."
    )
    stance: Literal["supports", "contradicts", "neutral"] = Field(
        description="How this evidence relates to the claim"
    )

    @field_validator("quote")
    @classmethod
    def quote_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Quote cannot be empty")
        return v.strip()


class AtomicClaim(BaseModel):
    """
    A single checkable sub-claim extracted from the user's input.

    Many real-world claims (especially WhatsApp forwards) contain multiple
    factual assertions. We break them apart to check each one independently.
    """

    claim_text: str = Field(description="A single, atomic, checkable factual assertion")
    is_checkable: bool = Field(
        description="False if this is an opinion, prediction, or value judgment "
        "(non-falsifiable claims return NOT_ENOUGH_EVIDENCE automatically)"
    )


class ClaimVerdict(BaseModel):
    """
    The structured output of the claim checker pipeline.
    This is what the LLM must return (via structured output / tool calling).
    """

    verdict: Verdict = Field(
        description="Overall verdict: SUPPORTED, CONTRADICTED, or NOT_ENOUGH_EVIDENCE"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict (0.0 = completely uncertain, 1.0 = certain). "
        "Should be low when evidence is sparse, old, or ambiguous.",
    )
    explanation: str = Field(
        description="2-3 sentence plain-language explanation of the verdict. "
        "Must mention which sources were used and why. "
        "MUST NOT state this as absolute truth — frame as 'according to retrieved articles'.",
        max_length=1000,
    )
    evidence: list[EvidenceSnippet] = Field(
        default_factory=list,
        description="List of cited evidence snippets. Each quote must be a verbatim excerpt.",
        max_length=10,
    )
    independent_sources: int = Field(
        ge=0,
        description="Number of INDEPENDENT sources corroborating the claim. "
        "Wire/syndication copies of the same story count as ONE source.",
    )
    freshness_note: str = Field(
        description="Comment on how recent the evidence is (e.g. 'Evidence from last 3 days')."
    )
    atomic_claims: list[AtomicClaim] = Field(
        default_factory=list,
        description="The input claim broken down into atomic, checkable assertions.",
    )


class ClaimCheckRequest(BaseModel):
    """API request model for POST /v1/check-claim"""

    claim: str = Field(
        min_length=10,
        max_length=2000,
        description="The claim, rumour, or forward to check",
        examples=["The government announced a 10% increase in fuel prices effective today."],
    )
    tier_filter: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="Restrict evidence search to a specific tier (1=official, 2=fact-checkers, 3=news)",
    )


class ClaimCheckResponse(BaseModel):
    """API response model for POST /v1/check-claim"""

    claim: str
    verdict: ClaimCheckRequest
    result: ClaimVerdict
    verified_evidence_count: int = Field(
        description="Number of evidence snippets that passed citation verification"
    )
    warning: str | None = Field(
        default=None,
        description="Optional warning (e.g. 'Evidence is more than 7 days old')",
    )
    disclaimer: str = Field(
        default=(
            "This verdict is based on articles retrieved from our news index "
            "(a rolling window of recent articles). It does not represent absolute truth. "
            "SUPPORTED means the retrieved articles support this claim, "
            "not that it is definitively true."
        )
    )
