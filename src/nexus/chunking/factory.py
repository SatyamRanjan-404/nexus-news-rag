"""
nexus.chunking.factory
~~~~~~~~~~~~~~~~~~~~~~
ChunkerFactory — creates the right chunker based on the config setting.

Usage:
    from nexus.chunking.factory import ChunkerFactory
    from nexus.config import get_settings

    chunker = ChunkerFactory.create(get_settings())
    chunks = chunker.chunk_many(articles)
"""

from __future__ import annotations

import logging

from nexus.chunking.base import BaseChunker
from nexus.chunking.recursive import RecursiveChunker
from nexus.config.settings import Settings

logger = logging.getLogger(__name__)


class ChunkerFactory:
    """
    Static factory that creates the appropriate BaseChunker implementation
    based on the CHUNKING_STRATEGY config value.

    Why a factory?
    The SemanticChunker requires the embedding model to already be loaded.
    The factory handles this dependency injection cleanly.
    """

    @staticmethod
    def create(settings: Settings, embedding_model=None) -> BaseChunker:
        """
        Create a chunker based on settings.

        Args:
            settings: Application settings (from get_settings()).
            embedding_model: Required only for strategy="semantic".
                             Should be the same LangChain Embeddings instance
                             used for the vector store (avoids loading twice).

        Returns:
            Configured BaseChunker instance.

        Raises:
            ValueError: If strategy is "semantic" but no embedding_model is provided.
        """
        strategy = settings.chunking_strategy

        if strategy == "recursive":
            chunker = RecursiveChunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            logger.info("Chunker: %r", chunker)
            return chunker

        elif strategy == "semantic":
            if embedding_model is None:
                raise ValueError(
                    "SemanticChunker requires an embedding_model instance. "
                    "Pass the loaded embedding model to ChunkerFactory.create()."
                )
            from nexus.chunking.semantic import SemanticChunker_

            chunker = SemanticChunker_(embedding_model=embedding_model)
            logger.info("Chunker: %r", chunker)
            return chunker

        else:
            raise ValueError(
                f"Unknown chunking strategy: '{strategy}'. "
                "Valid values: 'recursive', 'semantic'"
            )
