"""
LLM provider integrations for django-chain.
"""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

from django.conf import settings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

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
