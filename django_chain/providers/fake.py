"""
Fake integration for testing django-chain.
"""

from typing import Optional

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def get_fake_chat_model(responses: Optional[list[str]] = None, **kwargs) -> BaseChatModel:
    """
    Get a fake chat model instance for testing.

    Args:
        responses: List of responses to return in sequence
        **kwargs: Additional arguments for FakeListChatModel

    Returns:
        A configured FakeListChatModel instance
    """
    if responses is None:
        responses = ["This is a fake response."]
    return FakeListChatModel(responses=responses, **kwargs)


def get_fake_embedding_model(embedding_dim: int = 1536, **kwargs) -> Embeddings:
    """
    Get a fake embedding model instance for testing.

    Args:
        embedding_dim: Dimension of the fake embeddings
        **kwargs: Additional arguments for FakeEmbeddings

    Returns:
        A configured FakeEmbeddings instance
    """
    return FakeEmbeddings(size=embedding_dim, **kwargs)
