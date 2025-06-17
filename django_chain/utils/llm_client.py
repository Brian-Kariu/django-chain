import importlib
import logging

from django.conf import settings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# TODO: Add custom logging
LOGGER = logging.getLogger(__name__)


def create_llm_chat_client(provider: str, **kwargs) -> BaseChatModel | None:
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
    llm_configs = settings.DJANGO_LLM_SETTINGS.get("DEFAULT_CHAT_MODEL")
    model_name = llm_configs.get("name")
    model_temperature = llm_configs.get("temperature")
    model_max_tokens = llm_configs.get("max_tokens")
    api_key = llm_configs.get(f"{provider.upper()}_API_KEY")

    module_name = f"django_chain.providers.{provider}"
    client_function_name = f"get_{provider}_chat_model"
    try:
        llm_module = importlib.import_module(module_name)
        if hasattr(llm_module, client_function_name):
            dynamic_function = getattr(llm_module, client_function_name)
            return dynamic_function(
                api_key=api_key,
                model_name=model_name,
                temperature=model_temperature,
                max_tokens=model_max_tokens,
                **kwargs,
            )
        else:
            # TODO: Add specific test for this condition
            LOGGER.error(
                f"Chat function '{client_function_name}' not found in module '{module_name}'."
            )
    except ImportError as e:
        LOGGER.error(f"Error importing LLM Provider {module_name}: {e}")


def create_llm_embedding_client(provider: str, **kwargs) -> Embeddings | None:
    """
    Get an embedding model instance for the specified provider.
    #TODO: This function and the chat model are quite similar we can probably
    combine them but for easy readability they are separate.

    Args:
        provider: The embedding provider name (e.g., 'openai', 'google')
        **kwargs: Additional arguments for the embedding model

    Returns:
        A configured embedding model instance

    Raises:
        ImportError: If the required provider package is not installed
        ValueError: If the provider is not supported
    """
    module_name = f"django_chain.providers.{provider}"
    client_function_name = f"get_{provider}_embedding_model"
    try:
        llm_module = importlib.import_module(module_name)
        if hasattr(llm_module, client_function_name):
            dynamic_function = getattr(llm_module, client_function_name)
            return dynamic_function(**kwargs)
        else:
            # TODO: Add specific test for this condition
            LOGGER.error(
                f"Embedding function '{client_function_name}' not found in module '{module_name}'."
            )
    except ImportError as e:
        LOGGER.error(f"Error importing LLM Provider {module_name}: {e}")
