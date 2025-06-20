"""
OpenAI integration for django-chain.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


def get_openai_chat_model(
    api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7, **kwargs
) -> BaseChatModel:
    """
    Get an OpenAI chat model instance.

    Args:
        api_key: OpenAI API key
        model_name: Name of the model to use
        temperature: Model temperature (0.0 to 1.0)
        **kwargs: Additional arguments for ChatOpenAI

    Returns:
        A configured ChatOpenAI instance
    """
    return ChatOpenAI(
        model_name=model_name, temperature=temperature, openai_api_key=api_key, **kwargs
    )


def get_openai_embedding_model(
    api_key: str, model_name: str = "text-embedding-ada-002", **kwargs
) -> Embeddings:
    """
    Get an OpenAI embedding model instance.

    Args:
        api_key: OpenAI API key
        model_name: Name of the embedding model to use
        **kwargs: Additional arguments for OpenAIEmbeddings

    Returns:
        A configured OpenAIEmbeddings instance
    """
    return OpenAIEmbeddings(model=model_name, openai_api_key=api_key, **kwargs)
