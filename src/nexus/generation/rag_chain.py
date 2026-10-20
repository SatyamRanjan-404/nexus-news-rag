"""
nexus.generation.rag_chain
~~~~~~~~~~~~~~~~~~~~~~~~~~
LangChain LCEL RAG chain for grounded news question answering.

What is LCEL?
  LangChain Expression Language (LCEL) uses the pipe (|) operator to compose
  steps into a pipeline. Each step transforms the data and passes it to the next.
  Example: prompt | llm | parser

  Benefits:
  - Automatic streaming support (stream() works on any LCEL chain)
  - Easy to add/remove steps without restructuring code
  - Built-in async support

RAG Chain (Retrieval-Augmented Generation):
  1. Retrieve relevant chunks for the user's query
  2. Format them into a context block with source citations
  3. Build a prompt: system role + context + user question
  4. Call the LLM → stream the response back
  5. Return both the response stream AND the citations list

Grounding discipline (backs the resume claim "reducing hallucinated responses"):
  - The prompt explicitly instructs the LLM to ONLY use the provided context
  - It instructs the LLM to say "I cannot find..." rather than hallucinate
  - The LLM is told the date range of the context (freshness awareness)
  - Evaluation measures the "unsupported answer rate" (see Phase 8)

Abstention:
  If no context was retrieved (empty corpus or no relevant results), the chain
  returns None rather than calling the LLM with empty context. This prevents
  the LLM from generating a hallucinated response with no grounding.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Generator, Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from nexus.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are a factual news analyst. You answer questions by synthesising \
information from retrieved news articles.

STRICT RULES:
1. Base your answer ONLY on the NEWS CONTEXT provided below.
2. Do not use prior knowledge. Do not make up facts.
3. If the context does not contain the answer, say exactly:
   "I cannot find information about that in the retrieved news articles."
4. If answering, be specific: cite the source name and date when making claims.
5. Context articles were retrieved {date_note}.

NEWS CONTEXT:
{context}
"""

DATE_FORMAT = "%d %b %Y"


def format_context(chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    """
    Format retrieved chunks into a context block and a citations list.

    Args:
        chunks: Ranked, reranked chunks from the retrieval pipeline.

    Returns:
        (context_text, citations)
        - context_text: Formatted string to inject into the prompt
        - citations: List of {title, url, source, published} dicts for the UI
    """
    if not chunks:
        return "", []

    context_blocks = []
    citations = []
    seen_urls = set()

    for i, chunk in enumerate(chunks, start=1):
        block = (
            f"[Source {i}: {chunk.source_name} | {chunk.published_str[:10]}]\n"
            f"Title: {chunk.title}\n"
            f"{chunk.text}\n"
        )
        context_blocks.append(block)

        if chunk.url not in seen_urls:
            seen_urls.add(chunk.url)
            citations.append({
                "title": chunk.title,
                "link": chunk.url,
                "source": chunk.source_name,
                "published": chunk.published_str[:10],
            })

    context_text = "\n---\n".join(context_blocks)
    return context_text, citations


def get_date_note(chunks: list[RetrievedChunk]) -> str:
    """Build a freshness note about the retrieved context."""
    if not chunks:
        return "from an unknown time period"

    published_dates = [c.published_ts for c in chunks if c.published_ts > 0]
    if not published_dates:
        return "with unknown publication dates"

    newest_ts = max(published_dates)
    oldest_ts = min(published_dates)
    newest = datetime.fromtimestamp(newest_ts, tz=timezone.utc).strftime(DATE_FORMAT)
    oldest = datetime.fromtimestamp(oldest_ts, tz=timezone.utc).strftime(DATE_FORMAT)

    if oldest == newest:
        return f"from articles published on {newest}"
    return f"from articles published between {oldest} and {newest}"


# ── RAG Chain ─────────────────────────────────────────────────────


class RAGChain:
    """
    LangChain LCEL RAG chain for grounded news question answering.

    The chain is: prompt_template | llm | StrOutputParser

    This class wraps the chain with:
    - Context formatting from retrieved chunks
    - Citation extraction
    - Abstention when no context is available
    - Graceful error handling with retry on Groq API failures
    """

    def __init__(
        self,
        groq_api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
    ) -> None:
        """
        Args:
            groq_api_key: Groq API key.
            model: Groq model name for generation.
            temperature: LLM temperature (0.2 for slight creativity while staying grounded).
        """
        self._llm = ChatGroq(
            api_key=groq_api_key,
            model=model,
            temperature=temperature,
            streaming=True,  # enables .stream()
        )

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "{question}"),
        ])

        # The LCEL chain: prompt | llm | parser
        self._chain = self._prompt | self._llm | StrOutputParser()

    def stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[Iterator[str] | None, list[dict]]:
        """
        Generate a streamed answer and citations from retrieved chunks.

        Args:
            query: User's question.
            chunks: Retrieved and reranked chunks.

        Returns:
            (stream, citations) where:
            - stream: Iterator of text tokens (None if no context found)
            - citations: List of citation dicts for UI display
        """
        if not chunks:
            logger.info("No context available — abstaining")
            return None, []

        context, citations = format_context(chunks)
        date_note = get_date_note(chunks)

        if not context.strip():
            return None, []

        inputs = {
            "context": context,
            "date_note": date_note,
            "question": query,
        }

        try:
            stream = self._chain.stream(inputs)
            return stream, citations
        except Exception as exc:
            logger.error("RAG chain stream failed: %s", exc)
            return None, []

    def invoke(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str | None, list[dict]]:
        """
        Generate a non-streamed answer (useful for eval and claim checking).

        Args:
            query: User's question.
            chunks: Retrieved and reranked chunks.

        Returns:
            (answer_text, citations) — answer is None if no context.
        """
        if not chunks:
            return None, []

        context, citations = format_context(chunks)
        date_note = get_date_note(chunks)

        if not context.strip():
            return None, []

        inputs = {
            "context": context,
            "date_note": date_note,
            "question": query,
        }

        try:
            answer = self._chain.invoke(inputs)
            return answer, citations
        except Exception as exc:
            logger.error("RAG chain invoke failed: %s", exc)
            return None, []
