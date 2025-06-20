import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings.fake import FakeEmbeddings

from django_chain.providers.fake import get_fake_chat_model, get_fake_embedding_model


def test_get_fake_chat_model_default_response():
    model = get_fake_chat_model()
    assert isinstance(model, FakeListChatModel)
    assert model.responses == ["This is a fake response."]


def test_get_fake_chat_model_custom_responses():
    responses = ["Hi", "Hello", "Bye"]
    model = get_fake_chat_model(responses=responses)
    assert model.responses == responses


def test_get_fake_embedding_model_default_dim():
    model = get_fake_embedding_model()
    assert isinstance(model, FakeEmbeddings)
    assert model.size == 1536


def test_get_fake_embedding_model_custom_dim():
    model = get_fake_embedding_model(embedding_dim=512)
    assert model.size == 512
