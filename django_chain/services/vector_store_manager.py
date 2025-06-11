"""
Vector store manager service for django-chain.
"""

import logging
from typing import Any, Optional

from django.conf import settings

from django_chain.exceptions import MissingDependencyError, VectorStoreError
from django_chain.services.llm_client import LLMClient
from django_chain.vector_db_integrations import (
    add_documents,
    get_vector_store,
    retrieve_documents,
)

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
            vector_store_settings = settings.DJANGO_LLM_SETTINGS.get("VECTOR_STORE", {})
            store_type = store_type or vector_store_settings.get("TYPE", "pgvector")

            cache_key = f"{store_type}:{kwargs.get('collection_name', 'default')}"
            if cache_key in cls._instances:
                return cls._instances[cache_key]

            # Get embedding function
            embedding_function = LLMClient.get_embedding_model()

            # Get store configuration
            store_config = {
                "embedding_function": embedding_function,
                "collection_name": kwargs.get("collection_name")
                or vector_store_settings.get("PGVECTOR_COLLECTION_NAME", "langchain_documents"),
                **kwargs,
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
    def add_documents(
        cls,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        store_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            texts: List of text documents to add
            metadatas: Optional list of metadata dictionaries
            store_type: Optional store type override
            **kwargs: Additional arguments for the vector store

        Raises:
            VectorStoreError: If document addition fails
        """
        try:
            if not texts:
                raise ValueError("No texts provided to add to vector store")

            store = cls.get_vector_store(store_type=store_type, **kwargs)
            add_documents(store=store, texts=texts, metadatas=metadatas)
            logger.info(f"Successfully added {len(texts)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to add documents to vector store: {e!s}") from e

    @classmethod
    def retrieve_documents(
        cls, query: str, k: int = 4, store_type: Optional[str] = None, **kwargs
    ) -> list[dict[str, Any]]:
        """
        Retrieve documents from the vector store.

        Args:
            query: The search query
            k: Number of documents to retrieve
            store_type: Optional store type override
            **kwargs: Additional arguments for the vector store

        Returns:
            List of retrieved documents with their metadata

        Raises:
            VectorStoreError: If document retrieval fails
        """
        try:
            if not query:
                raise ValueError("No query provided for document retrieval")

            store = cls.get_vector_store(store_type=store_type, **kwargs)
            results = retrieve_documents(store=store, query=query, k=k)
            logger.info(f"Successfully retrieved {len(results)} documents from vector store")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents from vector store: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to retrieve documents from vector store: {e!s}") from e
