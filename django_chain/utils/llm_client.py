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

from django_chain.config import app_settings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from django_chain.utils import LOGGER
from django_chain.utils.memory_manager import get_chat_history


# Sensitive data patterns that should be redacted from logs
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "key",
    "secret",
    "password",
    "pwd",
    "access_token",
    "refresh_token",
    "auth_token",
    "bearer_token",
    "private_key",
    "secret_key",
    "client_secret",
    "auth",
    "authorization",
    "credentials",
    "credential",
    "openai_api_key",
    "google_api_key",
    "huggingface_api_key",
    "anthropic_api_key",
    "cohere_api_key",
}

# Patterns in values that indicate sensitive data
SENSITIVE_VALUE_PATTERNS = [
    "sk-",  # OpenAI API keys
    "AIza",  # Google API keys
    "hf_",  # HuggingFace API keys
    "pk-",  # Private keys
    "Bearer ",  # Bearer tokens
    "Basic ",  # Basic auth
]


def _sanitize_sensitive_data(data: Any, max_depth: int = 10) -> Any:
    """
    Recursively sanitize sensitive data from dictionaries, lists, and other data structures.

    Args:
        data: The data to sanitize
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Sanitized data with sensitive values redacted
    """
    if max_depth <= 0:
        return "[MAX_DEPTH_REACHED]"

    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            key_lower = str(key).lower().replace("-", "_").replace(" ", "_")

            is_sensitive = (
                key_lower in SENSITIVE_KEYS
                or any(key_lower.endswith(f"_{sensitive_key}") for sensitive_key in SENSITIVE_KEYS)
                or any(
                    key_lower.endswith(f"{sensitive_key}")
                    for sensitive_key in SENSITIVE_KEYS
                    if sensitive_key.endswith("_key")
                )
            )

            if is_sensitive:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = _sanitize_sensitive_data(value, max_depth - 1)

        return sanitized

    elif isinstance(data, (list, tuple)):
        return type(data)([_sanitize_sensitive_data(item, max_depth - 1) for item in data])

    elif isinstance(data, str):
        data_lower = data.lower()
        for pattern in SENSITIVE_VALUE_PATTERNS:
            if pattern.lower() in data_lower:
                return "[REDACTED]"
        return data

    else:
        return data


def _sanitize_model_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Specifically sanitize model parameters to ensure no API keys or sensitive data is logged.

    Args:
        params: Model parameters dictionary

    Returns:
        Sanitized parameters dictionary
    """
    if not isinstance(params, dict):
        return params

    sanitized_params = {}

    safe_params = {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "n",
        "stream",
        "model",
        "model_name",
        "max_retries",
        "timeout",
        "seed",
        "response_format",
    }

    for key, value in params.items():
        key_lower = str(key).lower().replace("-", "_").replace(" ", "_")

        is_sensitive = (
            key_lower in SENSITIVE_KEYS
            or any(key_lower.endswith(f"_{sensitive_key}") for sensitive_key in SENSITIVE_KEYS)
            or any(
                key_lower.endswith(f"{sensitive_key}")
                for sensitive_key in SENSITIVE_KEYS
                if sensitive_key.endswith("_key")
            )
        )

        if is_sensitive:
            sanitized_params[key] = "[REDACTED]"
        elif key in safe_params:
            sanitized_params[key] = value
        else:
            sanitized_params[key] = _sanitize_sensitive_data(value)

    return sanitized_params


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
    llm_configs = app_settings.DEFAULT_CHAT_MODEL
    model_name = llm_configs.get("name")
    model_temperature = llm_configs.get("temperature")
    model_max_tokens = llm_configs.get("max_tokens")
    api_key = llm_configs.get("api_key")

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
        return None
    return None


def create_llm_embedding_client(provider: str, **kwargs) -> Embeddings | None:
    """
    Get an embedding model instance for the specified provider.

    Args:
        provider: The LLM provider name (e.g., 'openai', 'google')
        **kwargs: Additional arguments for the embedding model

    Returns:
        A configured embedding model instance

    Raises:
        ImportError: If the required provider package is not installed
        ValueError: If the provider is not supported
    """
    embedding_configs = app_settings.DEFAULT_EMBEDDING_MODEL
    model_name = embedding_configs.get("name")
    api_key = embedding_configs.get("api_key", app_settings.DEFAULT_CHAT_MODEL.get("api_key"))

    module_name = f"django_chain.providers.{provider}"
    client_function_name = f"get_{provider}_embedding_model"
    try:
        llm_module = importlib.import_module(module_name)
        if hasattr(llm_module, client_function_name):
            dynamic_function = getattr(llm_module, client_function_name)
            return dynamic_function(
                api_key=api_key,
                model_name=model_name,
                **kwargs,
            )
    except ImportError as e:
        LOGGER.error(f"Error importing Embedding Provider {module_name}: {e}")
        return None
    return None


def _to_serializable(obj: Any) -> Any:
    """
    Convert objects to serializable format for JSON storage.

    Args:
        obj: Object to serialize

    Returns:
        Serializable representation of the object
    """
    if isinstance(obj, BaseMessage):
        return {
            "type": obj.__class__.__name__,
            "content": obj.content,
            "additional_kwargs": getattr(obj, "additional_kwargs", {}),
        }
    elif isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        try:
            serialized_dict = {key: _to_serializable(value) for key, value in obj.__dict__.items()}
            if not serialized_dict:
                return str(obj)
            return serialized_dict
        except Exception:
            return str(obj)
    else:
        return obj


class LoggingHandler(BaseCallbackHandler):
    """
    Enhanced callback handler for comprehensive LLM interaction logging.

    Captures detailed metadata including model information, token usage,
    performance metrics, and error details.
    """

    def __init__(self, interaction_log=None):
        super().__init__()
        self.interaction_log = interaction_log
        self.start_time = {}

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running."""
        run_id = kwargs.get("run_id")
        self.start_time[run_id] = time.time()
        if self.interaction_log:
            # Store the full kwargs structure as expected by tests
            prompt_data = {
                "prompts": prompts,
                "serialized": serialized,
                **{k: v for k, v in kwargs.items() if k not in ["run_id"]},
            }
            sanitized_prompt_data = _sanitize_sensitive_data(prompt_data)
            self.interaction_log.prompt_text = [_to_serializable(sanitized_prompt_data)]

            if serialized:
                sanitized_serialized = _sanitize_sensitive_data(serialized)

                model_name = (
                    sanitized_serialized.get("model_name")
                    or sanitized_serialized.get("model")
                    or sanitized_serialized.get("name")
                    or sanitized_serialized.get("id", {}).get("model_name")
                    if isinstance(sanitized_serialized.get("id"), dict)
                    else None or "unknown"
                )
                self.interaction_log.model_name = str(model_name)

                # Try different methods to extract provider information
                provider = sanitized_serialized.get("provider") or sanitized_serialized.get("_type")

                # If no direct provider info, try extracting from class names
                if not provider:
                    # Try extracting from nested class name
                    nested_class_name = (
                        sanitized_serialized.get("id", {}).get("lc", {}).get("name", "")
                    )
                    provider = self._extract_provider_from_class_name(nested_class_name)

                    # If that didn't work, try the direct name field
                    if provider == "unknown":
                        direct_name = sanitized_serialized.get("name", "")
                        provider = self._extract_provider_from_class_name(direct_name)

                # Final fallback
                if not provider or provider == "unknown":
                    provider = "unknown"

                self.interaction_log.provider = str(provider)

                model_params = {}
                for key in [
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "stop",
                ]:
                    if key in sanitized_serialized:
                        model_params[key] = sanitized_serialized[key]
                    elif "kwargs" in sanitized_serialized and key in sanitized_serialized["kwargs"]:
                        model_params[key] = sanitized_serialized["kwargs"][key]

                invocation_kwargs = kwargs.get("invocation_params", {})
                if invocation_kwargs:
                    sanitized_invocation = _sanitize_sensitive_data(invocation_kwargs)
                    model_params.update(sanitized_invocation)

                self.interaction_log.model_parameters = _sanitize_model_parameters(model_params)

            self.interaction_log.status = "PROCESSING"
            self.interaction_log.save()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        run_id = kwargs.get("run_id")
        if self.interaction_log and run_id in self.start_time:
            response_text = self._extract_response_text(response)
            self.interaction_log.response_text = response_text

            latency_seconds = time.time() - self.start_time[run_id]
            latency_ms = int(latency_seconds * 1000)
            # Clamp latency to reasonable bounds to prevent integer overflow
            # PostgreSQL integer range is -2147483648 to 2147483647
            self.interaction_log.latency_ms = max(0, min(latency_ms, 2147483647))

            token_usage = self._extract_token_usage(response, kwargs)
            if token_usage:
                self.interaction_log.input_tokens = token_usage.get("prompt_tokens", 0)
                self.interaction_log.output_tokens = token_usage.get("completion_tokens", 0)
            else:
                # Set to 0 when no token usage is found
                self.interaction_log.input_tokens = 0
                self.interaction_log.output_tokens = 0

            self.interaction_log.status = "success"
            self.interaction_log.save()

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when LLM errors."""
        run_id = kwargs.get("run_id")
        if self.interaction_log:
            self.interaction_log.error_message = str(error)
            self.interaction_log.status = "failure"

            if run_id in self.start_time:
                latency_seconds = time.time() - self.start_time[run_id]
                latency_ms = int(latency_seconds * 1000)
                # Clamp latency to reasonable bounds to prevent integer overflow
                # PostgreSQL integer range is -2147483648 to 2147483647
                self.interaction_log.latency_ms = max(0, min(latency_ms, 2147483647))

            token_usage = self._extract_token_usage(None, kwargs)
            if token_usage:
                self.interaction_log.input_tokens = token_usage.get("prompt_tokens", 0)
                self.interaction_log.output_tokens = token_usage.get("completion_tokens", 0)
            else:
                # Set to 0 when no token usage is found
                self.interaction_log.input_tokens = 0
                self.interaction_log.output_tokens = 0

        self.interaction_log.save()

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Run when a chat model starts running."""
        run_id = kwargs.get("run_id")
        self.start_time[run_id] = time.time()
        if self.interaction_log:
            # Store the full kwargs structure as expected by tests
            prompt_data = {
                "messages": messages,
                "serialized": serialized,
                **{k: v for k, v in kwargs.items() if k not in ["run_id"]},
            }
            sanitized_prompt_data = _sanitize_sensitive_data(prompt_data)
            self.interaction_log.prompt_text = [_to_serializable(sanitized_prompt_data)]

            if serialized:
                sanitized_serialized = _sanitize_sensitive_data(serialized)

                model_name = (
                    sanitized_serialized.get("model_name")
                    or sanitized_serialized.get("model")
                    or sanitized_serialized.get("name")
                    or "unknown"
                )
                self.interaction_log.model_name = str(model_name)

                # Try different methods to extract provider information
                provider = sanitized_serialized.get("provider") or sanitized_serialized.get("_type")

                # If no direct provider info, try extracting from class names
                if not provider:
                    # Try extracting from nested class name
                    nested_class_name = (
                        sanitized_serialized.get("id", {}).get("lc", {}).get("name", "")
                    )
                    provider = self._extract_provider_from_class_name(nested_class_name)

                    # If that didn't work, try the direct name field
                    if provider == "unknown":
                        direct_name = sanitized_serialized.get("name", "")
                        provider = self._extract_provider_from_class_name(direct_name)

                # Final fallback
                if not provider or provider == "unknown":
                    provider = "unknown"

                self.interaction_log.provider = str(provider)

                model_params = {}
                for key in [
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "stop",
                ]:
                    if key in sanitized_serialized:
                        model_params[key] = sanitized_serialized[key]

                invocation_kwargs = kwargs.get("invocation_params", {})
                if invocation_kwargs:
                    sanitized_invocation = _sanitize_sensitive_data(invocation_kwargs)
                    model_params.update(sanitized_invocation)

                self.interaction_log.model_parameters = _sanitize_model_parameters(model_params)

            self.interaction_log.status = "PROCESSING"
        self.interaction_log.save()

    def _extract_provider_from_class_name(self, class_name: str) -> str:
        """Extract provider name from LangChain class name."""
        if not class_name:
            return "unknown"

        class_name_lower = class_name.lower()
        if "openai" in class_name_lower:
            return "openai"
        elif "google" in class_name_lower or "gemini" in class_name_lower:
            return "google"
        elif "hugging" in class_name_lower:
            return "huggingface"
        elif "fake" in class_name_lower:
            return "fake"
        else:
            return "unknown"

    def _extract_response_text(self, response: Any) -> str:
        """Extract response text from various LangChain response formats."""
        if response is None:
            return ""

        if hasattr(response, "content"):
            return str(response.content)
        elif hasattr(response, "text"):
            return str(response.text)
        elif hasattr(response, "generations") and response.generations:
            if response.generations[0] and hasattr(response.generations[0][0], "text"):
                return response.generations[0][0].text
        elif isinstance(response, str):
            return response
        else:
            return _to_serializable(response)

    def _extract_token_usage(
        self, response: Any, kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, int]]:
        """Extract token usage information from response or kwargs."""
        if response and hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage")
            if token_usage:
                return {
                    "prompt_tokens": token_usage.get("prompt_tokens"),
                    "completion_tokens": token_usage.get("completion_tokens"),
                    "total_tokens": token_usage.get("total_tokens"),
                }

        if response and hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            return {
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        if "llm_output" in kwargs and kwargs["llm_output"]:
            token_usage = kwargs["llm_output"].get("token_usage")
            if token_usage:
                return {
                    "prompt_tokens": token_usage.get("prompt_tokens"),
                    "completion_tokens": token_usage.get("completion_tokens"),
                    "total_tokens": token_usage.get("total_tokens"),
                }

        return None


def _execute_and_log_workflow_step(
    workflow_chain,
    current_input: Dict[str, Any],
    execution_method: str = "invoke",
    execution_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Execute a workflow step with logging.

    Args:
        workflow_chain: The LangChain workflow to execute
        current_input: Input data for the workflow
        execution_method: Method to use for execution (invoke, stream, etc.)
        execution_config: Additional configuration for execution

    Returns:
        The result of the workflow execution
    """
    execution_config = execution_config or {}

    try:
        if execution_method.lower() == "invoke":
            return workflow_chain.invoke(current_input, config=execution_config)
        elif execution_method.lower() == "stream":
            return workflow_chain.stream(current_input, config=execution_config)
        elif execution_method.lower() == "batch":
            return workflow_chain.batch([current_input], config=execution_config)
        else:
            return workflow_chain.invoke(current_input, config=execution_config)
    except Exception as e:
        LOGGER.error(f"Error executing workflow: {e}", exc_info=True)
        raise


def add_wrapper_function(chain, function_name="runnable_with_message_history", **kwargs):
    """
    Add a wrapper function to a chain for message history support.

    Args:
        chain: The LangChain chain to wrap
        function_name: Name of the wrapper function
        **kwargs: Additional arguments for the wrapper

    Returns:
        The wrapped chain
    """
    memory_config = app_settings.MEMORY
    provider = memory_config.get("PROVIDER", "django")

    def get_session_history(session_id):
        return get_chat_history(session_id, provider=provider)

    WRAPPER_FUNCTIONS = {"runnable_with_message_history": RunnableWithMessageHistory}
    input_messages_key = kwargs.get("input_messages_key")
    history_messages_key = kwargs.get("history_messages_key")
    return WRAPPER_FUNCTIONS[function_name](
        chain,
        get_session_history=get_session_history,
        input_messages_key=input_messages_key,
        history_messages_key=history_messages_key,
    )
