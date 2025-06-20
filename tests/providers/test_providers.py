import pytest
from unittest.mock import patch, MagicMock
from django_chain.providers import get_chat_model, get_embedding_model


@pytest.mark.parametrize(
    "provider,import_path,function_name",
    [
        ("openai", "django_chain.providers.openai.get_openai_chat_model", "get_openai_chat_model"),
        ("google", "django_chain.providers.google.get_google_chat_model", "get_google_chat_model"),
        (
            "huggingface",
            "django_chain.providers.huggingface.get_huggingface_chat_model",
            "get_huggingface_chat_model",
        ),
        ("fake", "django_chain.providers.fake.get_fake_chat_model", "get_fake_chat_model"),
    ],
)
def test_get_chat_model(provider, import_path, function_name, settings):
    settings.DJANGO_LLM_SETTINGS = {
        "DEFAULT_LLM_PROVIDER": provider,
        "DEFAULT_CHAT_MODEL": {
            "name": "mock",
            "temperature": 0.5,
            f"{provider.upper()}_API_KEY": "test-key",
        },
    }

    with patch(import_path) as mock_func:
        mock_func.return_value = MagicMock(name=f"{provider}_chat_model")
        result = get_chat_model(provider, foo="bar", api_key="test")
        assert result == mock_func.return_value


def test_get_chat_model_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_chat_model("invalid")


@pytest.mark.parametrize(
    "provider,import_path,function_name",
    [
        (
            "openai",
            "django_chain.providers.openai.get_openai_embedding_model",
            "get_openai_embedding_model",
        ),
        (
            "google",
            "django_chain.providers.google.get_google_embedding_model",
            "get_google_embedding_model",
        ),
        (
            "huggingface",
            "django_chain.providers.huggingface.get_huggingface_embedding_model",
            "get_huggingface_embedding_model",
        ),
        (
            "fake",
            "django_chain.providers.fake.get_fake_embedding_model",
            "get_fake_embedding_model",
        ),
    ],
)
def test_get_embedding_model(provider, import_path, function_name):
    with patch(import_path) as mock_func:
        mock_func.return_value = MagicMock(name=f"{provider}_embed_model")
        result = get_embedding_model(provider, foo="bar")
        mock_func.assert_called_once_with(foo="bar")
        assert result == mock_func.return_value


def test_get_embedding_model_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embedding_model("invalid")
