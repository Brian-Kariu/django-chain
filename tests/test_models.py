import pytest
from model_bakery import baker
from django.core.exceptions import ValidationError

from django_chain.models import (
    Prompt,
    Workflow,
    ChatSession,
    ChatMessage,
    UserInteraction,
    InteractionLog,
)


@pytest.mark.django_db
class TestPrompt:
    def test_str_representation(self):
        prompt = baker.make(
            Prompt,
            name="TestPrompt",
            prompt_template={"langchain_type": "ChatPromptTemplate"},
            version=1,
            is_active=True,
        )
        assert str(prompt) == "TestPrompt v1 (Active)"

    def test_validation_disallows_multiple_active_prompts(self):
        baker.make(
            Prompt,
            name="Duplicate",
            prompt_template={"langchain_type": "ChatPromptTemplate"},
            is_active=True,
        )
        new_prompt = baker.prepare(Prompt, name="Duplicate", is_active=True)

        with pytest.raises(ValidationError) as e:
            new_prompt.full_clean()

    def test_prompt_template_requires_langchain_type(self):
        prompt = baker.prepare(Prompt, prompt_template={"not_langchain": "value"})
        with pytest.raises(ValidationError) as e:
            prompt.full_clean()


@pytest.mark.django_db
class TestWorkflow:
    def test_str_representation(self):
        workflow = baker.make(
            Workflow,
            name="Workflow1",
            workflow_definition=[
                {"type": "prompt", "name": "SimpleGreetingPrompt"},
                {"type": "llm", "config": {"temperature": 0.4}},
                {"type": "parser", "parser_type": "StrOutputParser"},
            ],
            is_active=True,
        )
        assert str(workflow) == "Workflow1 (Active)"

    def test_validation_disallows_multiple_active_with_same_name(self):
        baker.make(
            Workflow,
            name="TestFlow",
            workflow_definition=[
                {"type": "prompt", "name": "SimpleGreetingPrompt"},
                {"type": "llm", "config": {"temperature": 0.4}},
                {"type": "parser", "parser_type": "StrOutputParser"},
            ],
            is_active=True,
        )
        duplicate = baker.prepare(Workflow, name="TestFlow", is_active=True)

        with pytest.raises(ValidationError) as e:
            duplicate.full_clean()

    def test_workflow_definition_format_check(self):
        invalid = baker.prepare(Workflow, workflow_definition="not-a-list")
        with pytest.raises(ValidationError) as e:
            invalid.full_clean()


@pytest.mark.django_db
class TestChatSession:
    def test_index_fields(self):
        session = baker.make(ChatSession)
        assert session.session_id
        assert session.created_at
        assert session.updated_at

    def test_str_output(self):
        session = baker.make(ChatSession, title="MyChat")
        assert str(session) == "MyChat"


@pytest.mark.django_db
class TestChatMessage:
    def test_creation_and_string_output(self):
        session = baker.make(ChatSession)
        message = baker.make(ChatMessage, session=session, content="Hello world", role="USER")
        assert "Hello" in str(message)


@pytest.mark.django_db
class TestUserInteraction:
    def test_manager_methods(self):
        workflow = baker.make(
            Workflow,
            name="test_workflow",
            workflow_definition=[
                {"type": "prompt", "name": "SimpleGreetingPrompt"},
                {"type": "llm", "config": {"temperature": 0.4}},
                {"type": "parser", "parser_type": "StrOutputParser"},
            ],
        )
        interaction = UserInteraction.objects.create_for_workflow(
            workflow=workflow,
            input_data={"q": "What is AI?"},
            user_identifier="user123",
            session_id="d849cf3c-1d15-4791-b9d8-41e52cd940e4",
        )
        assert interaction.status == "processing"
        assert UserInteraction.objects.completed_interactions().count() == 0

        interaction.status = "success"
        interaction.save()

        assert UserInteraction.objects.completed_interactions().count() == 1

    def test_update_status_and_metrics(self):
        interaction = baker.make(UserInteraction)
        interaction.update_status_and_metrics(
            status="failure",
            llm_output={"message": "error"},
            total_cost_estimate=0.01,
            total_duration_ms=1500,
            error_message="Timeout",
        )
        assert interaction.status == "failure"
        assert interaction.error_message == "Timeout"


@pytest.mark.django_db
class TestInteractionLog:
    def test_str_output(self):
        id = 8392382
        model_name = "fake"
        status = "success"
        log = baker.make(InteractionLog, id=id, model_name=model_name, status=status)
        assert f"Log {id} - {model_name} ({status})" in str(log)


@pytest.mark.django_db
@pytest.mark.parametrize(
    "langchain_type,template,variables",
    [
        ("PromptTemplate", "Hello {name}", ["name"]),
        (
            "ChatPromptTemplate",
            [{"message_type": "human", "template": "Hi {x}", "input_variables": ["x"]}],
            ["x"],
        ),
    ],
)
def test_to_langchain_prompt_valid_cases(langchain_type, template, variables):
    template_data = {
        "langchain_type": langchain_type,
        "input_variables": variables,
    }
    if langchain_type == "PromptTemplate":
        template_data["template"] = template
    else:
        template_data["messages"] = template

    prompt = Prompt.objects.create(name=f"{langchain_type}-test", prompt_template=template_data)
    chain = prompt.to_langchain_prompt()
    assert chain is not None


@pytest.mark.django_db
def test_workflow_to_langchain_chain_prompt_only(monkeypatch):
    monkeypatch.setattr("django_chain.models.LANGCHAIN_AVAILABLE", True)
    prompt = baker.make(
        Prompt,
        name="Prompt1",
        is_active=True,
        prompt_template={
            "langchain_type": "PromptTemplate",
            "template": "Hello {name}",
            "input_variables": ["name"],
        },
    )
    workflow = baker.make(Workflow, workflow_definition=[{"type": "prompt", "name": prompt.name}])
    chain_no_logging = workflow.to_langchain_chain()
    assert chain_no_logging

    chain_with_logging = workflow.to_langchain_chain(log="true")
    assert chain_with_logging is not None


def test_workflow_llm_fake(monkeypatch, settings):
    monkeypatch.setattr("django_chain.models.LANGCHAIN_AVAILABLE", True)
    settings.DJANGO_LLM_SETTINGS = {
        "DEFAULT_LLM_PROVIDER": "fake",
        "DEFAULT_CHAT_MODEL": {
            "name": "mock",
            "temperature": 0.5,
            "FAKE_API_KEY": "test-key",
        },
    }
    global_config = {
        "DEFAULT_LLM_PROVIDER": "fake",
        "DEFAULT_CHAT_MODEL": {
            "name": "mock",
            "temperature": 0.5,
            "FAKE_API_KEY": "test-key",
        },
    }

    wf = baker.make(Workflow, workflow_definition=[{"type": "llm"}])
    assert wf.to_langchain_chain(llm_config=global_config)


def test_prompt_template_must_be_dict():
    prompt = baker.prepare(Prompt, prompt_template="not-a-dict")
    with pytest.raises(ValidationError) as e:
        prompt.full_clean()


def test_prompt_to_dict():
    p = baker.make(
        Prompt,
        name="sample prompt",
        prompt_template={
            "langchain_type": "PromptTemplate",
            "template": "Hello {name}",
            "input_variables": ["name"],
        },
    )
    d = p.to_dict()
    assert d["id"] == str(p.id)


def test_interactionlog_to_dict():
    p = baker.make(InteractionLog)
    d = p.to_dict()
    assert d["id"] == str(p.id)
