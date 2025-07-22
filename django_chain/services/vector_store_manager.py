"""
Vector store manager service for django-chain.
"""

import logging
from typing import Any, Optional

from django_chain.config import app_settings
from django_chain.exceptions import MissingDependencyError, VectorStoreError
from django_chain.utils.llm_client import create_llm_embedding_client
from django_chain.providers import get_vector_store

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Service for managing vector store operations."""

    _instances: dict[str, Any] = {}

    @classmethod
    def get_vector_store(cls, store_type: Optional[str] = None, **kwargs) -> Any:
        """
        Get a configured vector store instance.

        Args:
            store_type: Optional store type override
            **kwargs: Additional arguments for the vector store

        Returns:
            A configured vector store instance

        Raises:
            MissingDependencyError: If required store package is not installed
            VectorStoreError: If store configuration is invalid
        """
        try:
            vector_store_settings = app_settings.VECTOR_STORE
            store_type = store_type or vector_store_settings.get("TYPE", "pgvector")

            cache_key = f"{store_type}:{kwargs.get('collection_name', 'default')}"
            if cache_key in cls._instances:
                return cls._instances[cache_key]

            embed_config = app_settings.DEFAULT_EMBEDDING_MODEL
            provider = embed_config.get("provider", "fake")
            embedding_function = create_llm_embedding_client(provider)

            store_config = {
                "embedding_function": embedding_function,
                "collection_name": kwargs.get("collection_name")
                or vector_store_settings.get("PGVECTOR_COLLECTION_NAME", "langchain_documents"),
                **{k: v for k, v in kwargs.items() if k != "collection_name"},
            }

            store = get_vector_store(store_type=store_type, **store_config)
            cls._instances[cache_key] = store
            return store

        except ImportError as e:
            hint = f"Try running: pip install django-chain[{store_type}]"
            raise MissingDependencyError(
                f"Required vector store '{store_type}' is not installed.", hint=hint
            ) from e
        except Exception as e:
            logger.error(
                f"Error initializing vector store for type {store_type}: {e}",
                exc_info=True,
            )
            raise VectorStoreError(
                f"Failed to initialize vector store for type {store_type}: {e!s}"
            ) from e

    @classmethod
    def add_documents_to_store(
        cls,
        documents: list,
        store_type: Optional[str] = None,
        collection_name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Add documents to a vector store.

        Args:
            documents: List of documents to add
            store_type: Optional store type override
            collection_name: Optional collection name override
            **kwargs: Additional arguments

        Returns:
            Result of the add operation

        Raises:
            VectorStoreError: If adding documents fails
        """
        try:
            store = cls.get_vector_store(
                store_type=store_type, collection_name=collection_name, **kwargs
            )
            return store.add_texts(texts=documents)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to add documents: {e}") from e

    @classmethod
    def search_documents(
        cls,
        query: str,
        store_type: Optional[str] = None,
        collection_name: Optional[str] = None,
        k: int = 5,
        **kwargs,
    ) -> list:
        """
        Search for similar documents in the vector store.

        Args:
            query: Search query
            store_type: Optional store type override
            collection_name: Optional collection name override
            k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of similar documents

        Raises:
            VectorStoreError: If search fails
        """
        try:
            store = cls.get_vector_store(
                store_type=store_type, collection_name=collection_name, **kwargs
            )
            # Use vector store method directly
            return store.similarity_search(query=query, k=k)
        except Exception as e:
            logger.error(f"Error searching vector store: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to search documents: {e}") from e

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the vector store instance cache."""
        cls._instances.clear()
        logger.info("Vector store cache cleared")
