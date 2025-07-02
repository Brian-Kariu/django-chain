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
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import UUID

from django.conf import settings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

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
    workflow_chain, current_input: Any, execution_method: str, execution_config: dict = {}
) -> Any:
    """
    Executes a single step of the workflow, handles its logging, and returns its output.
    Uses _to_serializable for logging inputs/outputs.
    """
    EXECUTION_METHODS = {
        "INVOKE": workflow_chain.invoke,
        "BATCH": workflow_chain.batch,
        "BATCH_AS_COMPLETED": workflow_chain.batch_as_completed,
        "STREAM": workflow_chain.stream,
        "AINVOKE": workflow_chain.ainvoke,
        "ASTREAM": workflow_chain.astream,
        "ABATCH_AS_COMPLETED": workflow_chain.abatch_as_completed,
    }
    output = EXECUTION_METHODS[execution_method](current_input, config=execution_config)

    return output


class LoggingHandler(BaseCallbackHandler):
    def __init__(
        self,
        interaction_log,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.interaction_log = interaction_log
        self.start_times: Dict[UUID, float] = {}

    def _get_llm_model_name_from_serialized(self, serialized: Dict[str, Any]) -> str:
        """Helper to extract LLM model name from serialized data."""
        if "name" in serialized:
            return serialized["name"]
        if "kwargs" in serialized and "model_name" in serialized["kwargs"]:
            return serialized["kwargs"]["model_name"]
        if "kwargs" in serialized and "model" in serialized["kwargs"]:
            return serialized["kwargs"]["model"]
        return "unknown_llm_model"

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        self.start_times[run_id] = time.perf_counter()

        model_name = self._get_llm_model_name_from_serialized(serialized)

        self.interaction_log.provider = model_name
        self.interaction_log.prompt_text = ({"prompts": prompts},)
        self.interaction_log.metadata = (
            {
                "tags": tags,
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                **(metadata or {}),
            },
        )
        self.interaction_log.status = "processing"
        self.interaction_log.model_parameters = serialized.get("kwargs", {})
        self.interaction_log.input_tokens = 0
        self.interaction_log.output_tokens = 0

        self.interaction_log.save()

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when Chat Model starts running."""
        self.start_times[run_id] = time.perf_counter()

        model_name = self._get_llm_model_name_from_serialized(serialized)

        serializable_messages = []
        for msg_list in messages:
            serializable_messages.extend([msg.dict() for msg in msg_list])

        self.interaction_log.provider = model_name
        self.interaction_log.prompt_text = {"messages": serializable_messages}
        self.interaction_log.metadata = (
            {
                "tags": tags,
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                **(metadata or {}),
            },
        )
        self.interaction_log.status = "processing"
        self.interaction_log.model_parameters = serialized.get("kwargs", {})
        self.interaction_log.input_tokens = 0
        self.interaction_log.output_tokens = 0

        self.interaction_log.save()

    def on_llm_end(
        self,
        response,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        end_time = time.perf_counter()
        duration_ms = int((end_time - self.start_times.pop(run_id, end_time)) * 1000)

        output_text_parts = []
        prompt_tokens = 0
        completion_tokens = 0

        for generation_list in response.generations:
            for gen in generation_list:
                if hasattr(gen, "message") and hasattr(gen.message, "content"):
                    output_text_parts.append(gen.message.content)
                elif hasattr(gen, "text"):
                    output_text_parts.append(gen.text)

                if (
                    hasattr(gen, "message")
                    and hasattr(gen.message, "response_metadata")
                    and gen.message.response_metadata
                ):
                    usage = gen.message.response_metadata.get("usage", {})
                    prompt_tokens += usage.get("input_tokens", 0)  # Anthropic
                    completion_tokens += usage.get("output_tokens", 0)  # Anthropic
                    prompt_tokens += usage.get("prompt_tokens", 0)  # OpenAI
                    completion_tokens += usage.get("completion_tokens", 0)  # OpenAI
                elif hasattr(gen, "response_metadata") and gen.response_metadata:
                    token_usage = gen.response_metadata.get("token_usage", {})
                    prompt_tokens += token_usage.get("prompt_tokens", 0)
                    completion_tokens += token_usage.get("completion_tokens", 0)

        self.interaction_log.latency = duration_ms
        self.interaction_log.response_text = (
            {
                "text": "\n---\n".join(output_text_parts),
                "raw_response": response.dict(),
            },
        )
        self.interaction_log.input_tokens = prompt_tokens
        self.interaction_log.output_tokens = completion_tokens

        self.interaction_log.save()

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        end_time = time.perf_counter()
        duration_ms = int((end_time - self.start_times.pop(run_id, end_time)) * 1000)

        self.interaction_log.latency = duration_ms
        self.interaction_log.status = "failure"
        self.interaction_log.error_message = "error_message"

        self.interaction_log.save()
