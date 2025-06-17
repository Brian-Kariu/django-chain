from unittest.mock import MagicMock, patch
import pytest
from django_chain.providers.fake import BaseChatModel, FakeListChatModel
from langchain_community.embeddings.fake import FakeEmbeddings

from langchain_core.embeddings import Embeddings
from django_chain.utils.llm_client import create_llm_chat_client, create_llm_embedding_client


@pytest.mark.parametrize(
    "provider,input,expected",
    [
        ("fake", {}, FakeListChatModel(responses=["This is a fake response."])),
        ("fake", {"responses": ["test_response"]}, FakeListChatModel(responses=["test_response"])),
        ("fake", {"responses": "should fail"}, "Error"),
        ("incorrect", {}, "Error"),
    ],
    ids=[
        "Correct response no args",
        "Correct response without args",
        "Correct Provider with invalid args",
        "Invalid provider",
    ],
)
def test_create_chat_llm_client(provider, input, expected, caplog):
    if list(input.values()) == ["should fail"]:
        with pytest.raises(Exception):
            create_llm_chat_client(provider, **input)
    if provider == "incorrect":
        result = create_llm_chat_client(provider, **input)
        result = create_llm_chat_client(provider, **input)
        assert "Error importing LLM Provider" in caplog.text

    if provider == "fake" and expected != "Error":
        result = create_llm_chat_client(provider, **input)
        assert isinstance(result, BaseChatModel)
        assert result == expected


@pytest.mark.parametrize(
    "provider,input,expected",
    [
        ("fake", {}, FakeEmbeddings(size=1536)),
        ("fake", {"embedding_dim": 2000}, FakeEmbeddings(size=2000)),
        ("fake", {"embedding_dim": "should fail"}, "Error"),
        ("incorrect", {}, "Error"),
    ],
    ids=[
        "Correct response no args",
        "Correct response without args",
        "Correct Provider with invalid args",
        "Invalid provider",
    ],
)
def test_create_llm_embedding_client(provider, input, expected, caplog):
    if list(input.values()) == ["should fail"]:
        with pytest.raises(Exception):
            create_llm_embedding_client(provider, **input)
    if provider == "incorrect":
        result = create_llm_embedding_client(provider, **input)
        result = create_llm_embedding_client(provider, **input)
        assert "Error importing LLM Provider" in caplog.text

    if provider == "fake" and expected != "Error":
        result = create_llm_embedding_client(provider, **input)
        assert isinstance(result, Embeddings)
        assert result == expected
