import pytest
import types
from unittest.mock import patch, MagicMock
from django.conf import settings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from model_bakery import baker

from django_chain.models import Prompt, Workflow
from unittest.mock import MagicMock, patch
import pytest
from django_chain.providers.fake import BaseChatModel, FakeListChatModel
from langchain_community.embeddings.fake import FakeEmbeddings

from langchain_core.embeddings import Embeddings
from django_chain.utils.llm_client import (
    create_llm_chat_client,
    create_llm_embedding_client,
    _to_serializable,
    _execute_and_log_workflow_step,
    _get_langchain_prompt_object,
    create_langchain_workflow_chain,
)


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


@patch(
    "django_chain.utils.llm_client.importlib.import_module", side_effect=ImportError("no module")
)
def test_create_llm_chat_client_import_error(mock_import_module):
    settings.DJANGO_LLM_SETTINGS = {
        "DEFAULT_CHAT_MODEL": {"FAKE_API_KEY": "key", "name": "x", "temperature": 0.5}
    }
    client = create_llm_chat_client("fake")
    assert client is None


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


def test_to_serializable_ai_message():
    msg = AIMessage(content="Hi")
    result = _to_serializable(msg)
    assert isinstance(result, dict)
    assert result["content"] == "Hi"


def test_to_serializable_list_of_messages():
    msgs = [AIMessage(content="One"), AIMessage(content="Two")]
    result = _to_serializable(msgs)
    assert isinstance(result, list)
    assert all(isinstance(i, dict) for i in result)


def test_to_serializable_other_types():
    assert _to_serializable({"a": 1}) == {"a": 1}
    assert _to_serializable("text") == "text"
    assert _to_serializable(10) == 10
    assert _to_serializable(None) is None


def test_to_serializable_custom_object():
    class Foo:
        pass

    result = _to_serializable(Foo())
    assert isinstance(result, str)


@pytest.mark.django_db
def test_get_langchain_prompt_object(monkeypatch):
    monkeypatch.setattr("django_chain.utils.llm_client.Prompt", Prompt)
    prompt = baker.make(
        Prompt,
        name="TestPrompt",
        prompt_template={
            "langchain_type": "PromptTemplate",
            "template": "Hello {name}",
            "input_variables": ["name"],
        },
    )
    result = _get_langchain_prompt_object(prompt.name)
    assert result


@patch("django_chain.utils.llm_client._get_langchain_prompt_object")
@patch("django_chain.utils.llm_client.get_chat_model")
def test_create_workflow_chain_all_components(mock_get_chat, mock_get_prompt):
    mock_prompt = MagicMock()
    mock_get_prompt.return_value = mock_prompt

    mock_llm = MagicMock()
    mock_get_chat.return_value = mock_llm

    steps = [
        {"type": "prompt", "name": "PromptX"},
        {
            "type": "llm",
            "config": {
                "DEFAULT_LLM_PROVIDER": "fake",
                "DEFAULT_CHAT_MODEL": {"name": "m", "temperature": 0.3},
            },
        },
        {"type": "parser", "parser_type": "StrOutputParser"},
    ]
    config = {
        "DEFAULT_LLM_PROVIDER": "fake",
        "DEFAULT_CHAT_MODEL": {"name": "m", "temperature": 0.3},
    }

    chain = create_langchain_workflow_chain(steps, config)
    assert chain


def test_parser_type_invalid():
    steps = [{"type": "parser", "parser_type": "UnsupportedParser"}]
    config = {}
    with pytest.raises(ValueError, match="Unsupported parser type"):
        create_langchain_workflow_chain(steps, config)


@patch("django_chain.utils.llm_client.create_langchain_workflow_chain")
def test_execute_and_log_workflow_step(mock_create_chain):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"result": "ok"}
    mock_create_chain.return_value = mock_chain

    workflow = baker.make(
        Workflow,
        workflow_definition=[
            {"type": "prompt", "name": "SimpleGreetingPrompt"},
            {"type": "llm", "config": {"temperature": 0.4}},
            {"type": "parser", "parser_type": "StrOutputParser"},
        ],
    )
    output = _execute_and_log_workflow_step({"input": "hello"}, workflow, {})
    assert output == {"result": "ok"}
