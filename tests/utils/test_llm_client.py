from typing import List
from uuid import uuid4
import pytest
from unittest import TestCase
from unittest.mock import patch, MagicMock
from django.conf import settings
from langchain_core.messages import AIMessage

from unittest.mock import MagicMock, patch
import pytest
from django_chain.models import InteractionLog
from django_chain.providers.fake import BaseChatModel, FakeListChatModel
from langchain_community.embeddings.fake import FakeEmbeddings

from langchain_core.embeddings import Embeddings
from django_chain.utils.llm_client import (
    create_llm_chat_client,
    create_llm_embedding_client,
    _to_serializable,
    _execute_and_log_workflow_step,
    LoggingHandler,
)
from model_bakery import baker


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


@pytest.mark.parametrize(
    "input,execution_method,expected",
    [
        ({"input": "hello"}, "INVOKE", {"result": "ok"}),
        ({"input": "hello"}, "BATCH", {"result": "ok"}),
        ({"input": "hello"}, "BATCH_AS_COMPLETED", {"result": "ok"}),
        ({"input": "hello"}, "STREAM", {"result": "ok"}),
        ({"input": "hello"}, "AINVOKE", {"result": "ok"}),
        ({"input": "hello"}, "ASTREAM", {"result": "ok"}),
        ({"input": "hello"}, "ABATCH_AS_COMPLETED", {"result": "ok"}),
    ],
)
def test_execute_and_log_workflow_step(input, execution_method, expected):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = expected
    mock_chain.batch.return_value = expected
    mock_chain.batch_as_completed.return_value = expected
    mock_chain.stream.return_value = expected
    mock_chain.ainvoke.return_value = expected
    mock_chain.astream.return_value = expected
    mock_chain.abatch_as_completed.return_value = expected

    output = _execute_and_log_workflow_step(
        workflow_chain=mock_chain, current_input=input, execution_method=execution_method
    )
    assert output == expected


class MockGeneration:
    def __init__(
        self,
        text=None,
        message=None,
        response_metadata=None,
    ):
        self.text = text
        self.message = message
        self.response_metadata = response_metadata


class MockLLMResponse:
    def __init__(self, generations: List[List[MockGeneration]]):
        """
        Mock constructor for an LLM response object.
        It holds a list of lists of generations.
        """
        self.generations = generations

    def dict(self):
        """
        Mock method to serialize the response to a dictionary,
        used for the 'raw_response' field in the log.
        """
        serialized_generations = []
        for gen_list in self.generations:
            inner_list = []
            for gen in gen_list:
                if gen.text:
                    inner_list.append({"text": gen.text})
                elif gen.message:
                    inner_list.append({"message": gen.message.dict()})
            serialized_generations.append(inner_list)
        return {"generations": serialized_generations}


class MockBaseMessage:
    def dict(self):
        return "this is a real message"


@pytest.mark.django_db
class TestLoggingHandler:
    def setup_method(self):
        self.log = baker.make(InteractionLog)
        self.instance = LoggingHandler(interaction_log=self.log)

    @pytest.mark.parametrize(
        "serialized,expected",
        [
            ({"name": {"name": "fake"}}, {"name": "fake"}),
            ({"kwargs": {"model_name": {"name": "fake"}}}, {"name": "fake"}),
            ({"kwargs": {"model": {"name": "fake"}}}, {"name": "fake"}),
        ],
    )
    def test_get_llm_model_name(self, serialized, expected):
        result = self.instance._get_llm_model_name_from_serialized(serialized)
        assert result == expected

    @pytest.mark.parametrize(
        "input",
        [
            {
                "run_id": 1,
                "parent_id": 2,
                "tags": ["custom"],
                "prompts": ["custom prompt"],
                "serialized": {"name": "fake"},
                "metadata": {"custom": "extra"},
            },
        ],
    )
    def test_on_llm_start(self, input):
        self.instance.on_llm_start(**input)
        assert self.log.provider == "fake"
        assert len(self.log.prompt_text) == 1
        assert self.log.prompt_text[0]["prompts"][0] == "custom prompt"

    @pytest.mark.parametrize(
        "input",
        [
            {
                "run_id": 1,
                "parent_id": 2,
                "tags": ["custom"],
                "messages": [[MockBaseMessage()]],
                "serialized": {"name": "fake"},
                "metadata": {"custom": "extra"},
            },
        ],
    )
    def test_on_chat_start(self, input):
        self.instance.on_chat_model_start(**input)
        assert self.log.provider == "fake"
        assert len(self.log.prompt_text) == 1

    def test_on_llm_end(self):
        run_id = uuid4()
        self.instance.start_times[run_id] = 2.0

        response = MockLLMResponse(
            generations=[
                [MockGeneration(text="This is the first part of the response.")],
                [MockGeneration(text="And this is the second part, separated by '---'.")],
            ]
        )
        self.instance.on_llm_end(response=response, run_id=run_id)

        assert self.log.input_tokens == 0
        assert self.log.output_tokens == 0
        assert self.log.status == "success"

    def test_on_llm_error(self):
        run_id = uuid4()
        self.instance.start_times[run_id] = 12.0

        error = Exception("A simulated LLM error occurred.")

        self.instance.on_llm_error(error=error, run_id=run_id)

        assert self.log.status == "failure"
        assert self.log.error_message == "error_message"
