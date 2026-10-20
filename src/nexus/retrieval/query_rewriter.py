"""
nexus.retrieval.query_rewriter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Query rewriting for follow-up questions in multi-turn conversations.

Problem:
  In a multi-turn conversation:
    User: "What did the RBI announce today?"
    AI: "RBI announced a rate cut to 6.5%..."
    User: "What was the market reaction?"

  The second question "What was the market reaction?" is ambiguous without
  context. If passed directly to retrieval, it will find articles about any
  market reaction, not specifically the one to the RBI rate cut.

Solution:
  Before retrieval, pass the conversation history + current question to a
  lightweight LLM call that rewrites the question as a standalone query:
    "What was the market reaction to the RBI interest rate cut announcement?"

  This rewritten query retrieves far more relevant results.

Design:
  - Only activates when conversation history has ≥ 1 prior turn
  - Uses llama-3.1-8b-instant (fast, cheap) not the full 70b model
  - Temperature 0 (deterministic rewriting)
  - Bounded to last 3 turns of history (enough context, avoids long prompts)
  - If the rewriting call fails, falls back to the original query gracefully
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """You are a query rewriting assistant. Your only job is to rewrite a 
follow-up question into a complete, self-contained search query.

Rules:
- Make the question understandable without the conversation history
- Preserve all specific entities (names, numbers, dates, organisations)
- Do NOT answer the question, only rewrite it
- Return ONLY the rewritten query, nothing else
- If the question is already self-contained, return it unchanged"""


class QueryRewriter:
    """
    Rewrites follow-up questions into standalone queries using a lightweight LLM.
    """

    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant") -> None:
        """
        Args:
            groq_api_key: Groq API key.
            model: Model to use for rewriting (use a fast/cheap model).
        """
        self._llm = ChatGroq(
            api_key=groq_api_key,
            model=model,
            temperature=0.0,
            max_tokens=200,
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    def rewrite(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
        max_history_turns: int = 3,
    ) -> str:
        """
        Rewrite a query into a standalone form given conversation context.

        Args:
            query: The user's current question.
            conversation_history: List of {"role": "user"|"assistant", "content": str} dicts.
            max_history_turns: Maximum prior turns to include as context.

        Returns:
            Rewritten standalone query string. Falls back to original on error.
        """
        if not conversation_history:
            return query  # no history = already standalone

        # Trim to last N turns
        recent_history = conversation_history[-max_history_turns * 2 :]

        # Format history for the prompt
        history_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content'][:300]}"
            for turn in recent_history
        )

        user_message = (
            f"Conversation history:\n{history_text}\n\n"
            f"Follow-up question: {query}\n\n"
            f"Rewritten standalone query:"
        )

        try:
            response = self._llm.invoke([
                SystemMessage(content=REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ])
            rewritten = response.content.strip()
            if rewritten:
                logger.debug("Query rewritten: '%s' → '%s'", query[:60], rewritten[:60])
                return rewritten
        except Exception as exc:
            logger.warning("Query rewriting failed (falling back to original): %s", exc)

        return query


def build_claim_queries(claim: str, groq_api_key: str) -> list[str]:
    """
    Generate 1-3 search queries from a claim for the claim checker.

    Different queries target different aspects of the claim:
    - Factual query: rephrase as a fact to verify
    - Entity query: focus on key entities (people, organisations, numbers)
    - Negation check: look for contradicting evidence

    Args:
        claim: The claim text to check.
        groq_api_key: Groq API key.

    Returns:
        List of 1-3 search query strings.
    """
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=300,
    )

    prompt = f"""You are a fact-checking query generator. Given a claim, generate 2-3 
search queries to find evidence that would support or contradict it.

Rules:
- Focus on verifiable facts (entities, numbers, dates, events)
- Include an entity-focused query (extract key names/organisations/numbers)
- Make queries specific and short (5-12 words each)
- Return ONLY the queries, one per line, no numbering or bullets

Claim: {claim}

Search queries:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
        queries = lines[:3]  # cap at 3 queries
        if not queries:
            return [claim]
        return queries
    except Exception as exc:
        logger.warning("Claim query generation failed: %s", exc)
        return [claim]  # fallback: use claim directly as query
