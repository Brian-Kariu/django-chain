"""
LLM provider integrations for django-chain.

This package provides functions to instantiate chat and embedding models for supported LLM providers
(OpenAI, Google, HuggingFace, Fake) and acts as a central registry for provider selection.
"""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

from django.conf import settings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from django_chain.exceptions import MissingDependencyError


def get_chat_model(provider: str, **kwargs) -> BaseChatModel:
    """
    Get a chat model instance for the specified provider.

    Args:
        provider: The LLM provider name (e.g., 'openai', 'google')
        **kwargs: Additional arguments for the chat model

    Returns:
        A configured chat model instance

    Raises:
        ImportError: If the required provider package is not installed
        ValueError: If the provider is not supported
    """
    if provider == "openai":
        from django_chain.providers.openai import get_openai_chat_model

        return get_openai_chat_model(**kwargs)
    elif provider == "google":
        from django_chain.providers.google import get_google_chat_model

        return get_google_chat_model(**kwargs)
    elif provider == "huggingface":
        from django_chain.providers.huggingface import get_huggingface_chat_model

        return get_huggingface_chat_model(**kwargs)
    elif provider == "fake":
        from django_chain.providers.fake import get_fake_chat_model

        return get_fake_chat_model(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_embedding_model(provider: str, **kwargs) -> Embeddings:
    """
    Get an embedding model instance for the specified provider.

    Args:
        provider: The embedding provider name (e.g., 'openai', 'google')
        **kwargs: Additional arguments for the embedding model

    Returns:
        A configured embedding model instance

    Raises:
        ImportError: If the required provider package is not installed
        ValueError: If the provider is not supported
    """
    if provider == "openai":
        from django_chain.providers.openai import get_openai_embedding_model

        return get_openai_embedding_model(**kwargs)
    elif provider == "google":
        from django_chain.providers.google import get_google_embedding_model

        return get_google_embedding_model(**kwargs)
    elif provider == "huggingface":
        from django_chain.providers.huggingface import get_huggingface_embedding_model

        return get_huggingface_embedding_model(**kwargs)
    elif provider == "fake":
        from django_chain.providers.fake import get_fake_embedding_model

        return get_fake_embedding_model(**kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def get_vector_store(store_type: str, embedding_function: Embeddings, **kwargs) -> VectorStore:
    """
    Get a vector store instance for the specified type.

    Args:
        store_type: The vector store type (e.g., 'pgvector', 'chroma')
        embedding_function: The embedding function to use
        **kwargs: Additional arguments for the vector store

    Returns:
        A configured vector store instance

    Raises:
        ImportError: If the required store package is not installed
        ValueError: If the store type is not supported
        NotImplementedError: If the store type is not yet implemented
    """
    if store_type == "pgvector":
        from django_chain.providers.pgvector import get_pgvector_store

        return get_pgvector_store(embedding_function=embedding_function, **kwargs)
    elif store_type == "chroma":
        # TODO: Implement chroma provider
        raise NotImplementedError("Chroma vector store is not yet implemented")
    elif store_type == "pinecone":
        # TODO: Implement pinecone provider
        raise NotImplementedError("Pinecone vector store is not yet implemented")
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
