"""
LLM client utilities for django-chain.

This module provides functions to instantiate chat and embedding models, serialize LangChain objects,
and build workflow chains for LLM-powered Django applications.

Typical usage example:
    chat_model = create_llm_chat_client("openai", ...)
    embedding_model = create_llm_embedding_client("openai", ...)
    chain = create_langchain_workflow_chain([...], {...})
"""

import importlib
import logging
import os
from typing import Any

from django.conf import settings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from django_chain.models import Prompt
from django_chain.providers import get_chat_model

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


def _to_serializable(obj: Any) -> Any:
    """
    Converts LangChain objects (like BaseMessage) and other non-serializable types
    into JSON-compatible dictionaries or strings.
    """
    if isinstance(obj, BaseMessage):
        return obj.dict()
    elif isinstance(obj, list) and all(isinstance(item, BaseMessage) for item in obj):
        return [item.dict() for item in obj]
    elif isinstance(obj, (dict, list, str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _execute_and_log_workflow_step(
    current_input: Any,
    workflow_record: Any,
    global_llm_config: dict,
) -> Any:
    """
    Executes a single step of the workflow, handles its logging, and returns its output.
    Uses _to_serializable for logging inputs/outputs.
    """
    workflow_chain = create_langchain_workflow_chain(
        workflow_record.workflow_definition, global_llm_config
    )
    output = workflow_chain.invoke(current_input)
    return output


def _get_langchain_prompt_object(prompt_name: dict):  # noqa: C901
    """
    Internal utility to convert a prompt_data dictionary into a LangChain prompt object.
    This should be similar to Prompt.to_langchain_prompt() but independent of the DB model.
    """
    prompt = Prompt.objects.get(name=prompt_name)
    return prompt.to_langchain_prompt()


def create_langchain_workflow_chain(workflow_definition: list, llm_config: dict):  # noqa: C901
    """
    Constructs and returns a LangChain RunnableSequence from a workflow definition
    and a global LLM configuration. This function does NOT query the database
    or read Django settings.

    Args:
        workflow_definition (list): A JSON-like list of dictionaries defining the workflow steps.
                                    Example: [{"type": "prompt", "prompt_data": {"template": "...", "langchain_type": "PromptTemplate"}},
                                              {"type": "llm", "config": {"model_name": "gpt-4"}},
                                              {"type": "parser", "parser_type": "StrOutputParser"}]
        llm_config (dict): A dictionary containing the default LLM configuration, e.g.,
                           {"provider": "openai", "model_name": "gpt-3.5-turbo", "temperature": 0.7, "api_key": "..."}.
                           This is effectively the DJANGO_LLM settings passed in as an argument.

    Returns:
        langchain_core.runnables.RunnableSequence: The constructed LangChain chain.

    Raises:
        ValueError: If the workflow_definition is invalid or components cannot be instantiated.
        ImportError: If required LangChain libraries are not installed.
    """
    chain_components = []
    for i, step_data in enumerate(workflow_definition):
        step_type = step_data.get("type")

        prompt_name = step_data.get("name")
        if step_type == "prompt":
            try:
                chain_components.append(_get_langchain_prompt_object(prompt_name))
            except Exception as e:
                raise ValueError(f"Workflow step {i}: Error creating prompt object: {e}")

        elif step_type == "llm":
            llm_config_override = step_data.get("config", {})
            current_llm_config = {
                **llm_config,
                **llm_config_override,
            }

            llm_provider = current_llm_config.get("DEFAULT_LLM_PROVIDER")
            model_name = current_llm_config["DEFAULT_CHAT_MODEL"]["name"]
            temperature = current_llm_config["DEFAULT_CHAT_MODEL"]["temperature"]
            api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")

            if not llm_provider or not model_name:
                raise ValueError(
                    f"Workflow step {i}: LLM step requires 'llm_provider' and 'model_name' in its config or global LLM config."
                )

            llm_instance = get_chat_model(llm_provider, temperature=temperature, api_key=api_key)

            chain_components.append(llm_instance)

        elif step_type == "parser":
            parser_type = step_data.get("parser_type")
            parser_args = step_data.get("parser_args", {})

            if not parser_type:
                raise ValueError(f"Workflow step {i}: Parser step requires 'parser_type'.")

            parser_instance = None
            if parser_type == "StrOutputParser":
                parser_instance = StrOutputParser(**parser_args)
            elif parser_type == "JsonOutputParser":
                parser_instance = JsonOutputParser(**parser_args)
            else:
                raise ValueError(f"Workflow step {i}: Unsupported parser type: {parser_type}")

            if parser_instance:
                chain_components.append(parser_instance)

        else:
            raise ValueError(f"Workflow step {i}: Unknown component type: {step_type}")

    if not chain_components:
        raise ValueError("Workflow definition is empty or contains no valid components.")

    from functools import reduce

    return reduce(lambda a, b: a | b, chain_components)
