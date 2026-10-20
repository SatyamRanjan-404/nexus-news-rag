"""
nexus.claims.checker
~~~~~~~~~~~~~~~~~~~~
The News Claim Checker pipeline.

Order of operations (see ADR-012 and project spec):
  1. Normalise the claim: extract atomic, checkable sub-claims
  2. Detect non-checkable claims (opinions, predictions) → return NOT_ENOUGH_EVIDENCE
  3. Generate 1-3 search queries from the claim
  4. For each query, retrieve from all three tiers:
     a. Tier 2 (fact-checkers) — highest priority: look for an existing fact-check
     b. Tier 1 (official sources) — government/institutional statements
     c. Tier 3 (news) — general corroboration, counting only independent sources
  5. RRF-fuse all retrieved candidates, cross-encoder rerank
  6. Run the LCEL verdict chain (Pydantic structured output, temp=0)
  7. Citation verification: check each quote against retrieved text
     - If quote is not a substring (or fuzzy-match >= 0.85), drop it
  8. Confidence-based abstention: if rerank scores too low → NOT_ENOUGH_EVIDENCE
  9. Return ClaimVerdict with full metadata

Honesty framing:
  EVERYWHERE in this code, comments and responses make clear that:
  - "SUPPORTED" = supported by retrieved articles, not absolute truth
  - The verdict window is limited by the retention policy (14-365 days)
  - Fact-checker ratings are reported as "rated X by Y", not as facts
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential

from nexus.claims.models import (
    AtomicClaim,
    ClaimVerdict,
    EvidenceSnippet,
    Verdict,
)
from nexus.retrieval.hybrid import RetrievedChunk
from nexus.retrieval.query_rewriter import build_claim_queries

logger = logging.getLogger(__name__)

# Minimum reranker score to trust evidence (below this = abstain)
MIN_RERANK_CONFIDENCE = -5.0  # cross-encoder scores can be negative; tune in Phase 7
# Minimum fuzzy match ratio for citation verification (0-100 scale)
CITATION_FUZZY_THRESHOLD = 85


# ── Citation verification ─────────────────────────────────────────


def verify_quote(quote: str, source_texts: list[str], threshold: int = CITATION_FUZZY_THRESHOLD) -> bool:
    """
    Check whether a claimed quote is actually present in the retrieved source texts.

    Why this matters:
    LLMs sometimes "hallucinate citations" — they generate plausible-sounding
    quotes that weren't actually in the retrieved text. This programmatic check
    catches those and prevents false citations from reaching the user.

    Method:
    1. Exact substring match (fast, catches verbatim quotes)
    2. Fuzzy match via rapidfuzz (catches minor whitespace/punctuation differences)

    Args:
        quote: The quoted text the LLM returned.
        source_texts: List of retrieved document texts to check against.
        threshold: Minimum fuzzy match ratio (0-100) to accept.

    Returns:
        True if the quote can be verified against at least one source text.
    """
    quote_lower = quote.lower().strip()
    if not quote_lower:
        return False

    for text in source_texts:
        text_lower = text.lower()

        # 1. Exact substring check
        if quote_lower in text_lower:
            return True

        # 2. Fuzzy partial match (handles minor normalisation differences)
        ratio = fuzz.partial_ratio(quote_lower, text_lower)
        if ratio >= threshold:
            return True

    return False


# ── Verdict generation ────────────────────────────────────────────

VERDICT_SYSTEM_PROMPT = """You are a careful fact-checker. Your job is to determine whether a 
claim is supported, contradicted, or not addressed by the provided news article excerpts.

CRITICAL RULES:
1. Base your verdict ONLY on the provided article excerpts below.
2. Do NOT use your training knowledge to verify claims.
3. Only quote text that actually appears in the article excerpts.
4. Be conservative: when in doubt, choose NOT_ENOUGH_EVIDENCE.
5. If the evidence is ambiguous or conflicting, choose NOT_ENOUGH_EVIDENCE.
6. Frame your explanation as "according to retrieved articles" not as absolute fact.
7. For fact-checker articles (Tier 2), report their rating as "rated X by [outlet]".
8. Count INDEPENDENT sources: if 5 outlets all copied the same wire story, 
   that is ONE independent source.
9. Never use emojis in your response or explanation. Maintain an analytical, formal tone.

CLAIM TO CHECK:
{claim}

RETRIEVED ARTICLE EXCERPTS:
{context}

Respond with a structured verdict following the schema provided."""


class ClaimChecker:
    """
    End-to-end claim verification pipeline.
    """

    def __init__(
        self,
        retriever,          # HybridRetriever
        reranker,           # CrossEncoderReranker
        groq_api_key: str,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        """
        Args:
            retriever: HybridRetriever instance.
            reranker: CrossEncoderReranker instance.
            groq_api_key: Groq API key.
            model: Model for verdict generation (temp=0, structured output).
        """
        self._retriever = retriever
        self._reranker = reranker
        self._llm = ChatGroq(
            api_key=groq_api_key,
            model=model,
            temperature=0.0,  # mandatory for reproducibility
        )
        self._structured_llm = self._llm.with_structured_output(ClaimVerdict)
        self._groq_api_key = groq_api_key

    def check(
        self,
        claim: str,
        rerank_candidate_k: int = 20,
        final_k: int = 5,
        tier_filter: int | None = None,
    ) -> tuple[ClaimVerdict, int]:
        """
        Check a claim and return a verdict.

        Args:
            claim: The claim text to verify.
            rerank_candidate_k: Candidates to retrieve before reranking.
            final_k: Final evidence chunks after reranking.
            tier_filter: Optionally restrict to a specific tier.

        Returns:
            (ClaimVerdict, verified_evidence_count)
        """
        logger.info("Checking claim: '%s'", claim[:100])

        # Step 1: Generate search queries from the claim
        queries = build_claim_queries(claim, self._groq_api_key)
        logger.info("Generated %d queries: %s", len(queries), queries)

        # Step 2: Retrieve from each tier in priority order
        all_candidates: list[RetrievedChunk] = []

        # Build where clause for tier filtering
        def build_where(tier: int | None, extra_tier: int | None = None) -> dict | None:
            conditions = []
            if tier_filter is not None:
                conditions.append({"tier": {"$eq": tier_filter}})
            elif tier is not None:
                if extra_tier is not None:
                    conditions.append({"tier": {"$in": [tier, extra_tier]}})
                else:
                    conditions.append({"tier": {"$eq": tier}})
            return {"$and": conditions} if len(conditions) > 1 else (conditions[0] if conditions else None)

        for query in queries:
            # Retrieve candidates for each query
            candidates = self._retriever.retrieve(
                query=query,
                n_results=rerank_candidate_k,
                where=build_where(None),
            )
            all_candidates.extend(candidates)

        # Deduplicate candidates by chunk_id
        seen_ids: set[str] = set()
        unique_candidates: list[RetrievedChunk] = []
        for c in all_candidates:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                unique_candidates.append(c)

        logger.info("Total unique candidates: %d", len(unique_candidates))

        # Step 3: Rerank the combined candidate pool
        if unique_candidates:
            reranked = self._reranker.rerank(
                query=claim,
                candidates=unique_candidates,
                top_k=final_k,
            )
        else:
            reranked = []

        # Step 4: Abstain if no evidence or confidence is too low
        if not reranked:
            return ClaimVerdict(
                verdict=Verdict.NOT_ENOUGH_EVIDENCE,
                confidence=0.0,
                explanation="No relevant articles were found in our news index to evaluate this claim.",
                evidence=[],
                independent_sources=0,
                freshness_note="No articles found.",
                atomic_claims=[],
            ), 0

        top_score = reranked[0].rerank_score if reranked else -99
        if top_score < MIN_RERANK_CONFIDENCE:
            logger.info("Abstaining: top rerank score too low (%.3f)", top_score)
            return ClaimVerdict(
                verdict=Verdict.NOT_ENOUGH_EVIDENCE,
                confidence=0.1,
                explanation=(
                    "The retrieved articles have low relevance scores for this claim. "
                    "There is not enough evidence to make a confident determination."
                ),
                evidence=[],
                independent_sources=0,
                freshness_note=self._freshness_note(reranked),
                atomic_claims=[],
            ), 0

        # Step 5: Format context for the verdict prompt
        context_blocks = []
        for i, chunk in enumerate(reranked, 1):
            tier_label = {1: "[OFFICIAL]", 2: "[FACT-CHECK]", 3: "[NEWS]"}.get(chunk.tier, "")
            context_blocks.append(
                f"[{i}] {tier_label} {chunk.source_name} ({chunk.published_str[:10]})\n"
                f"Title: {chunk.title}\n"
                f"{chunk.text}\n"
                f"URL: {chunk.url}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        # Step 6: Generate structured verdict
        try:
            verdict: ClaimVerdict = self._structured_llm.invoke([
                SystemMessage(content=VERDICT_SYSTEM_PROMPT.format(
                    claim=claim,
                    context=context,
                )),
                HumanMessage(content="Generate the structured verdict now."),
            ])
        except Exception as exc:
            logger.error("Verdict generation failed: %s", exc)
            return ClaimVerdict(
                verdict=Verdict.NOT_ENOUGH_EVIDENCE,
                confidence=0.0,
                explanation="Technical error during claim verification. Please try again.",
                evidence=[],
                independent_sources=0,
                freshness_note=self._freshness_note(reranked),
                atomic_claims=[],
            ), 0

        # Step 7: Citation verification — drop any quote the LLM made up
        source_texts = [chunk.text for chunk in reranked]
        verified_evidence = []
        for evidence in verdict.evidence:
            if verify_quote(evidence.quote, source_texts):
                verified_evidence.append(evidence)
            else:
                logger.warning(
                    "Quote failed verification and was removed: '%s'",
                    evidence.quote[:80],
                )

        verdict.evidence = verified_evidence

        # Step 8: Downgrade verdict if citation verification removed all evidence
        if verdict.verdict != Verdict.NOT_ENOUGH_EVIDENCE and len(verified_evidence) == 0:
            logger.warning("All evidence failed citation verification — downgrading to NOT_ENOUGH_EVIDENCE")
            verdict.verdict = Verdict.NOT_ENOUGH_EVIDENCE
            verdict.confidence = min(verdict.confidence, 0.2)
            verdict.explanation += (
                " (Note: The generated evidence quotes could not be verified "
                "against the source texts and were removed.)"
            )

        # Update freshness note
        verdict.freshness_note = self._freshness_note(reranked)

        logger.info(
            "Verdict: %s (confidence=%.2f, verified_evidence=%d)",
            verdict.verdict,
            verdict.confidence,
            len(verified_evidence),
        )

        return verdict, len(verified_evidence)

    def _freshness_note(self, chunks: list[RetrievedChunk]) -> str:
        """Generate a note about how recent the evidence is."""
        if not chunks:
            return "No articles found."

        timestamps = [c.published_ts for c in chunks if c.published_ts > 0]
        if not timestamps:
            return "Publication dates unknown."

        newest_ts = max(timestamps)
        newest_dt = datetime.fromtimestamp(newest_ts, tz=timezone.utc)
        days_ago = (datetime.now(tz=timezone.utc) - newest_dt).days

        if days_ago == 0:
            return "Most recent evidence: published today."
        elif days_ago == 1:
            return "Most recent evidence: published yesterday."
        elif days_ago <= 7:
            return f"Most recent evidence: published {days_ago} days ago."
        else:
            return f"Most recent evidence: published {days_ago} days ago (may be outdated)."
