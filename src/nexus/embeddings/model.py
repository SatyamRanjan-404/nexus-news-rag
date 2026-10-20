"""
nexus.embeddings.model
~~~~~~~~~~~~~~~~~~~~~~
Pluggable embedding model wrapper.

Why a wrapper instead of using HuggingFaceEmbeddings directly?
  1. Caching: the model is loaded once per process (expensive: ~200MB, ~5s)
  2. Experimentation: swap models via config without touching other code
  3. Batching: the wrapper handles batch encoding for large document sets
  4. Type safety: consistent interface regardless of which model is used

Current supported models (via sentence-transformers / HuggingFace):
  - all-MiniLM-L6-v2 (default): 22M params, 384 dims, ~90MB, very fast on CPU
  - BAAI/bge-small-en-v1.5: 33M params, 384 dims, ~130MB, often better recall
  - both produce 384-dimensional vectors

Interview answer on model choice:
  "I chose all-MiniLM-L6-v2 as the baseline because it's fast on CPU and
   well-studied. For the experiments, I also tested bge-small-en-v1.5 which
   consistently ranks higher on MTEB benchmarks. The eval showed [X]% Recall@5
   improvement with bge-small at [Y]ms extra latency per query — we chose
   [winner] based on that tradeoff."
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=3)  # cache up to 3 different model configs simultaneously
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Load and return a HuggingFace embedding model (cached per model name).

    The model is downloaded from HuggingFace Hub on first call (~90-130MB)
    and cached locally by sentence-transformers in ~/.cache/huggingface/.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        LangChain-compatible HuggingFaceEmbeddings instance.
    """
    logger.info("Loading embedding model: %s", model_name)
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # explicit CPU — no CUDA dependency
        encode_kwargs={
            "normalize_embeddings": True,  # unit-norm vectors → cosine similarity = dot product
            "batch_size": 64,
        },
    )
    logger.info("Embedding model loaded: %s", model_name)
    return model


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    """
    Embed a list of text strings.

    Args:
        texts: List of strings to embed.
        model_name: Model to use.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    model = get_embedding_model(model_name)
    return model.embed_documents(texts)


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    """
    Embed a single query string.

    Args:
        query: Query text.
        model_name: Model to use.

    Returns:
        Embedding vector.
    """
    model = get_embedding_model(model_name)
    return model.embed_query(query)
