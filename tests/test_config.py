"""
Streamlined tests for django-chain configuration system.
"""

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from django_chain.config import app_settings, validate_settings, get_setting, get_provider_api_key


@pytest.mark.django_db
class TestConfiguration:
    """Test configuration system."""

    def test_default_settings_loaded(self):
        """Test that default settings are properly loaded."""
        assert app_settings.DEFAULT_LLM_PROVIDER == "fake"
        assert app_settings.DEFAULT_CHAT_MODEL["name"] == "fake-model"
        assert app_settings.VECTOR_STORE["TYPE"] == "pgvector"

    def test_get_method_functionality(self):
        """Test the get() method works correctly."""
        assert app_settings.get("DEFAULT_LLM_PROVIDER") == "fake"
        assert app_settings.get("NONEXISTENT_SETTING") is None
        assert app_settings.get("NONEXISTENT_SETTING", "default") == "default"

    def test_invalid_setting_raises_error(self):
        """Test that accessing invalid settings raises AttributeError."""
        with pytest.raises(AttributeError, match="Invalid django-chain setting"):
            _ = app_settings.INVALID_SETTING

    @pytest.mark.parametrize(
        "provider,model_name,temperature",
        [
            ("openai", "gpt-4", 0.5),
            ("google", "gemini-pro", 0.3),
            ("fake", "test-model", 0.7),
        ],
    )
    @override_settings(DJANGO_LLM_SETTINGS={})
    def test_settings_override(self, provider, model_name, temperature):
        """Test that user settings override defaults."""
        with override_settings(
            DJANGO_LLM_SETTINGS={
                "DEFAULT_LLM_PROVIDER": provider,
                "DEFAULT_CHAT_MODEL": {
                    "name": model_name,
                    "temperature": temperature,
                    "api_key": "test-key",
                },
            }
        ):
            app_settings.reload()
            assert app_settings.DEFAULT_LLM_PROVIDER == provider
            assert app_settings.DEFAULT_CHAT_MODEL["name"] == model_name
            assert app_settings.DEFAULT_CHAT_MODEL["temperature"] == temperature

    @override_settings(
        DJANGO_LLM_SETTINGS={
            "DEFAULT_LLM_PROVIDER": "openai",
            "DEFAULT_CHAT_MODEL": {
                "name": "gpt-3.5-turbo",
                "temperature": 0.5,
                "api_key": "test",
                "max_tokens": 1024,
            },
        }
    )
    def test_settings_validation(self):
        """Test settings validation functionality."""
        validate_settings()

    @pytest.mark.parametrize(
        "provider,expected_key",
        [
            ("fake", "FAKE_API_KEY"),
        ],
    )
    def test_provider_api_key_retrieval(self, provider, expected_key):
        """Test API key retrieval for different providers."""
        with override_settings(**{expected_key: "test-api-key"}):
            key = get_provider_api_key(provider)
            assert key == "FAKE_API_KEY"

    @override_settings(
        DJANGO_LLM_SETTINGS={
            "DEFAULT_CHAT_MODEL": {"temperature": 0.5},
            "VECTOR_STORE": {"PGVECTOR_COLLECTION_NAME": "custom_collection"},
        }
    )
    def test_deep_merge_functionality(self):
        """Test that nested settings are properly merged."""
        assert app_settings.DEFAULT_CHAT_MODEL["temperature"] == 0.5
        assert app_settings.VECTOR_STORE["PGVECTOR_COLLECTION_NAME"] == "custom_collection"
