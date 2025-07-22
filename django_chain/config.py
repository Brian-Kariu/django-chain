"""
Configuration management for django-chain.

This module provides a Django App Settings pattern similar to Django REST Framework.
Settings are loaded from Django's settings.DJANGO_LLM_SETTINGS with sensible defaults.
"""

import warnings
from typing import Any
from typing import Dict

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.test.signals import setting_changed

DEFAULTS = {
    "DEFAULT_LLM_PROVIDER": "fake",
    "DEFAULT_CHAT_MODEL": {
        "name": "fake-model",
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_key": "FAKE_API_KEY",
    },
    "DEFAULT_EMBEDDING_MODEL": {
        "provider": "fake",
        "name": "fake-embedding",
    },
    "VECTOR_STORE": {
        "TYPE": "pgvector",
        "PGVECTOR_COLLECTION_NAME": "langchain_documents",
    },
    "ENABLE_LLM_LOGGING": True,
    "LLM_LOGGING_LEVEL": "DEBUG",
    "MEMORY": {
        "PROVIDER": "django",
        "DEFAULT_TYPE": "buffer",
        "WINDOW_SIZE": 5,
    },
    "CHAIN": {
        "DEFAULT_OUTPUT_PARSER": "str",
        "ENABLE_MEMORY": True,
    },
    "CACHE_LLM_SETTINGS": {
        "CACHE_LLM_RESPONSES": False,
        "CACHE_TTL_SECONDS": 3600,
    },
}

DEPRECATED_SETTINGS = {
    "CACHING": (
        "CACHE_LLM_SETTINGS",
        "The 'CACHING' setting has been renamed to 'CACHE_LLM_SETTINGS'.",
    ),
}

VALID_PROVIDERS = ["openai", "google", "huggingface", "fake"]
VALID_OUTPUT_PARSERS = ["str", "json"]
VALID_MEMORY_TYPES = ["buffer", "buffer_window", "summary"]
VALID_VECTOR_STORE_TYPES = ["pgvector", "chroma", "pinecone"]
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
VALID_MEMORY_PROVIDERS = ["django", "inmemory"]


class AppSettings:
    """
    A settings object that allows access to django-chain settings,
    checking for user settings first, then falling back to defaults.

    This pattern is inspired by Django REST Framework's settings.py
    https://github.com/encode/django-rest-framework/blob/master/rest_framework/settings.py
    """

    def __init__(self, defaults: Dict[str, Any] | None = None):
        self.defaults = defaults or DEFAULTS
        self._cached_attrs = set()

    @property
    def user_settings(self) -> Dict[str, Any]:
        if not hasattr(self, "_user_settings"):
            self._user_settings = getattr(settings, "DJANGO_LLM_SETTINGS", {})
        return self._user_settings

    def __getattr__(self, attr: str) -> Any:
        if attr in DEPRECATED_SETTINGS:
            new_name, deprecation_warning = DEPRECATED_SETTINGS[attr]
            warnings.warn(
                f"{deprecation_warning} Please use '{new_name}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self, new_name)

        if attr not in self.defaults:
            raise AttributeError(f"Invalid django-chain setting: '{attr}'")

        try:
            val = self.user_settings[attr]
        except KeyError:
            val = self.defaults[attr]

        self._cached_attrs.add(attr)
        setattr(self, attr, val)
        return val

    def reload(self) -> None:
        """
        Reload settings, clearing any cached values.
        Useful for testing when settings change.
        """
        for attr in self._cached_attrs:
            delattr(self, attr)
        self._cached_attrs.clear()
        if hasattr(self, "_user_settings"):
            delattr(self, "_user_settings")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value, with optional default.
        Similar to dict.get() behavior.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def update(self, **kwargs) -> None:
        """
        Update settings values. Useful for testing.
        """
        if not hasattr(self, "_user_settings"):
            self._user_settings = getattr(settings, "DJANGO_LLM_SETTINGS", {}).copy()
        else:
            self._user_settings = self._user_settings.copy()

        self._user_settings.update(kwargs)

        for attr in list(self._cached_attrs):
            if attr in kwargs:
                delattr(self, attr)
                self._cached_attrs.remove(attr)
                getattr(self, attr)

    def validate(self) -> None:
        """
        Validate all settings configuration.
        Raises ImproperlyConfigured for invalid settings.
        """
        self._validate_provider_settings()
        self._validate_chat_model_settings()
        self._validate_embedding_model_settings()
        self._validate_vector_store_settings()
        self._validate_memory_settings()
        self._validate_chain_settings()
        self._validate_logging_settings()
        self._validate_cache_settings()

    def _validate_provider_settings(self) -> None:
        """Validate LLM provider configuration."""
        provider = self.DEFAULT_LLM_PROVIDER
        if provider not in VALID_PROVIDERS:
            raise ImproperlyConfigured(
                f"Invalid DEFAULT_LLM_PROVIDER: '{provider}'. "
                f"Must be one of: {', '.join(VALID_PROVIDERS)}"
            )

    def _validate_chat_model_settings(self) -> None:
        """Validate chat model configuration."""
        chat_model = self.DEFAULT_CHAT_MODEL
        if not isinstance(chat_model, dict):
            raise ImproperlyConfigured("DEFAULT_CHAT_MODEL must be a dictionary")

        required_keys = ["name", "temperature", "max_tokens"]
        for key in required_keys:
            if key not in chat_model:
                raise ImproperlyConfigured(f"DEFAULT_CHAT_MODEL missing required key: '{key}'")

        temperature = chat_model.get("temperature")
        if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 2.0):
            raise ImproperlyConfigured(
                f"DEFAULT_CHAT_MODEL temperature must be a number between 0.0 and 2.0, got: {temperature}"
            )

        max_tokens = chat_model.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ImproperlyConfigured(
                f"DEFAULT_CHAT_MODEL max_tokens must be a positive integer, got: {max_tokens}"
            )

        provider = self.DEFAULT_LLM_PROVIDER
        if provider != "fake":
            api_key = chat_model.get("api_key")
            if not api_key or api_key == "FAKE_API_KEY":
                raise ImproperlyConfigured(
                    f"DEFAULT_CHAT_MODEL api_key is required for provider '{provider}'. "
                    f"Please set a valid API key or use provider 'fake' for testing."
                )

    def _validate_embedding_model_settings(self) -> None:
        """Validate embedding model configuration."""
        embedding_model = self.DEFAULT_EMBEDDING_MODEL
        if not isinstance(embedding_model, dict):
            raise ImproperlyConfigured("DEFAULT_EMBEDDING_MODEL must be a dictionary")

        required_keys = ["provider", "name"]
        for key in required_keys:
            if key not in embedding_model:
                raise ImproperlyConfigured(f"DEFAULT_EMBEDDING_MODEL missing required key: '{key}'")

        provider = embedding_model.get("provider")
        if provider not in VALID_PROVIDERS:
            raise ImproperlyConfigured(
                f"Invalid DEFAULT_EMBEDDING_MODEL provider: '{provider}'. "
                f"Must be one of: {', '.join(VALID_PROVIDERS)}"
            )

    def _validate_vector_store_settings(self) -> None:
        """Validate vector store configuration."""
        vector_store = self.VECTOR_STORE
        if not isinstance(vector_store, dict):
            raise ImproperlyConfigured("VECTOR_STORE must be a dictionary")

        store_type = vector_store.get("TYPE")
        if store_type and store_type not in VALID_VECTOR_STORE_TYPES:
            raise ImproperlyConfigured(
                f"Invalid VECTOR_STORE TYPE: '{store_type}'. "
                f"Must be one of: {', '.join(VALID_VECTOR_STORE_TYPES)}"
            )

        if store_type == "pgvector":
            collection_name = vector_store.get("PGVECTOR_COLLECTION_NAME")
            if not collection_name:
                raise ImproperlyConfigured(
                    "VECTOR_STORE PGVECTOR_COLLECTION_NAME is required when TYPE is 'pgvector'"
                )

    def _validate_memory_settings(self) -> None:
        """Validate memory configuration."""
        memory = self.MEMORY
        if not isinstance(memory, dict):
            raise ImproperlyConfigured("MEMORY must be a dictionary")

        memory_type = memory.get("DEFAULT_TYPE")
        if memory_type and memory_type not in VALID_MEMORY_TYPES:
            raise ImproperlyConfigured(
                f"Invalid MEMORY DEFAULT_TYPE: '{memory_type}'. "
                f"Must be one of: {', '.join(VALID_MEMORY_TYPES)}"
            )

        provider = memory.get("PROVIDER")
        if provider and provider not in VALID_MEMORY_PROVIDERS:
            raise ImproperlyConfigured(
                f"Invalid MEMORY PROVIDER: '{provider}'. Must be one of: {', '.join(VALID_MEMORY_PROVIDERS)}"
            )

        if memory_type == "buffer_window":
            window_size = memory.get("WINDOW_SIZE")
            if not isinstance(window_size, int) or window_size <= 0:
                raise ImproperlyConfigured(
                    f"MEMORY WINDOW_SIZE must be a positive integer when DEFAULT_TYPE is 'buffer_window', got: {window_size}"
                )

    def _validate_chain_settings(self) -> None:
        """Validate chain configuration."""
        chain = self.CHAIN
        if not isinstance(chain, dict):
            raise ImproperlyConfigured("CHAIN must be a dictionary")

        output_parser = chain.get("DEFAULT_OUTPUT_PARSER")
        if output_parser and output_parser not in VALID_OUTPUT_PARSERS:
            raise ImproperlyConfigured(
                f"Invalid CHAIN DEFAULT_OUTPUT_PARSER: '{output_parser}'. "
                f"Must be one of: {', '.join(VALID_OUTPUT_PARSERS)}"
            )

    def _validate_logging_settings(self) -> None:
        """Validate logging configuration."""
        log_level = self.LLM_LOGGING_LEVEL
        if log_level and log_level not in VALID_LOG_LEVELS:
            raise ImproperlyConfigured(
                f"Invalid LLM_LOGGING_LEVEL: '{log_level}'. "
                f"Must be one of: {', '.join(VALID_LOG_LEVELS)}"
            )

    def _validate_cache_settings(self) -> None:
        """Validate cache configuration."""
        cache_settings = self.CACHE_LLM_SETTINGS
        if not isinstance(cache_settings, dict):
            raise ImproperlyConfigured("CACHE_LLM_SETTINGS must be a dictionary")

        ttl = cache_settings.get("CACHE_TTL_SECONDS")
        if ttl is not None and (not isinstance(ttl, int) or ttl <= 0):
            raise ImproperlyConfigured(
                f"CACHE_LLM_SETTINGS CACHE_TTL_SECONDS must be a positive integer, got: {ttl}"
            )


app_settings = AppSettings(DEFAULTS)


def reload_app_settings(*args, **kwargs) -> None:
    """
    Reload settings when Django settings change.
    Connected to Django's setting_changed signal.
    """
    setting = kwargs["setting"]
    if setting == "DJANGO_LLM_SETTINGS":
        app_settings.reload()


setting_changed.connect(reload_app_settings)


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a django-chain setting value.

    Args:
        key: The setting name
        default: Default value if setting not found

    Returns:
        The setting value or default
    """
    return app_settings.get(key, default)


def validate_settings() -> None:
    """
    Validate all django-chain settings configuration.
    Can be called in AppConfig.ready() for early validation.

    Raises:
        ImproperlyConfigured: If any settings are invalid
    """
    app_settings.validate()


def get_provider_api_key(provider: str) -> str:
    """
    Get API key for a specific provider.

    Args:
        provider: The provider name (e.g., 'openai', 'google')

    Returns:
        The API key for the provider

    Raises:
        ImproperlyConfigured: If API key is not configured for non-fake providers
    """
    if provider == "fake":
        return "FAKE_API_KEY"

    chat_model = app_settings.DEFAULT_CHAT_MODEL
    api_key = chat_model.get("api_key")

    fake_keys = ["FAKE_API_KEY", "fake key", "fake_key", "test_key", "test-key"]
    if api_key and api_key.lower() not in [k.lower() for k in fake_keys]:
        return api_key

    provider_key = f"{provider.upper()}_API_KEY"
    api_key = app_settings.get(provider_key)

    if not api_key or api_key.lower() in [k.lower() for k in fake_keys]:
        raise ImproperlyConfigured(
            f"API key for provider '{provider}' is not configured. "
            f"Please set it in DEFAULT_CHAT_MODEL.api_key or {provider_key}"
        )

    return api_key
