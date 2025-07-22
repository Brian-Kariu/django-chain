import pytest
from model_bakery import baker
from django.core.exceptions import ValidationError
from django_chain.exceptions import PromptValidationError, WorkflowValidationError
from django_chain.models import (
    Prompt,
    Workflow,
    ChatSession,
    ChatHistory,
    InteractionLog,
)


@pytest.mark.django_db
class TestVersionedModels:
    """Test versioned model behavior (Prompt, Workflow)."""

    @pytest.mark.parametrize(
        "model_class,template_field,template_value",
        [
            (
                Prompt,
                "prompt_template",
                {
                    "langchain_type": "ChatPromptTemplate",
                    "messages": [
                        {"role": "system", "content": "You're an assistant who's good at ability"},
                        {"MessagesPlaceholder": "history"},
                        {"role": "human", "content": "question"},
                    ],
                },
            ),
            (Workflow, "workflow_definition", [{"type": "prompt"}, {"type": "llm"}]),
        ],
    )
    def test_str_representation(self, model_class, template_field, template_value):
        """Test string representation for versioned models."""
        kwargs = {
            "name": "TestModel",
            template_field: template_value,
            "version": 1,
            "is_active": True,
        }
        instance = baker.make(model_class, **kwargs)
        assert str(instance) == "TestModel v1 (Active)"

    @pytest.mark.parametrize(
        "model_class,template_field,template_value",
        [
            (
                Prompt,
                "prompt_template",
                {
                    "langchain_type": "ChatPromptTemplate",
                    "messages": [
                        {"role": "system", "content": "You're an assistant who's good at ability"},
                        {"MessagesPlaceholder": "history"},
                        {"role": "human", "content": "question"},
                    ],
                },
            ),
            (Workflow, "workflow_definition", [{"type": "prompt"}, {"type": "llm"}]),
        ],
    )
    def test_unique_active_constraint(self, model_class, template_field, template_value):
        """Test that only one active instance per name is allowed."""
        kwargs = {
            "name": "Duplicate",
            template_field: template_value,
            "is_active": True,
        }
        baker.make(model_class, **kwargs)

        duplicate = baker.prepare(model_class, name="Duplicate", is_active=True)
        with pytest.raises(ValidationError):
            duplicate.full_clean()

    @pytest.mark.parametrize(
        "model_class,invalid_field,invalid_value",
        [
            (Prompt, "prompt_template", {"not_langchain": "value"}),
            (Prompt, "prompt_template", "not-a-dict"),
            (Workflow, "workflow_definition", "not-a-list"),
        ],
    )
    def test_validation_errors(self, model_class, invalid_field, invalid_value):
        """Test validation errors for invalid field values."""
        kwargs = {invalid_field: invalid_value}
        instance = baker.prepare(model_class, **kwargs)
        if model_class == Prompt:
            with pytest.raises(PromptValidationError):
                instance.full_clean()
        elif model_class == Workflow:
            with pytest.raises(WorkflowValidationError):
                instance.full_clean()


@pytest.mark.django_db
class TestChatModels:
    """Test chat-related models."""

    def test_chat_session_uuid_generation(self):
        """Test that session_id is automatically generated as UUID."""
        session = baker.make(ChatSession)
        import uuid

        assert isinstance(session.session_id, uuid.UUID)

        session2 = baker.make(ChatSession)
        assert session.session_id != session2.session_id

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("MyChat", "MyChat"),
            ("", None),
            (None, None),
        ],
    )
    def test_chat_session_str_representation(self, title, expected):
        """Test ChatSession string representation."""
        session = baker.make(ChatSession, title=title)
        if expected:
            assert str(session) == expected
        else:
            assert str(session) == f"Chat Session {session.session_id}"

    def test_chat_session_archive_activate(self):
        """Test chat session archive/activate functionality."""
        session = baker.make(ChatSession, is_active=True)

        session.archive()
        assert not session.is_active

        session.activate()
        assert session.is_active

    def test_chat_history_creation(self):
        """Test ChatHistory model functionality."""
        session = baker.make(ChatSession)
        message = baker.make(ChatHistory, session=session, content="Hello world", role="USER")

        assert "Hello" in str(message)
        assert message.session == session
        assert not message.is_deleted

        message.soft_delete()
        assert message.is_deleted

        message.restore()
        assert not message.is_deleted


@pytest.mark.django_db
class TestInteractionLog:
    """Test InteractionLog model."""

    def test_str_representation(self):
        """Test InteractionLog string representation."""
        log = baker.make(InteractionLog, model_name="gpt-4", status="SUCCESS")
        assert "gpt-4" in str(log)
        assert "SUCCESS" in str(log)

    def test_to_dict_sanitization(self):
        """Test that to_dict method sanitizes sensitive data."""
        log = baker.make(
            InteractionLog,
            model_parameters={"api_key": "secret", "temperature": 0.7},
            metadata={"user_data": "sensitive"},
        )

        result = log.to_dict()
        assert "id" in result
        assert str(log.id) == result["id"]
        assert result["model_parameters"]["api_key"] == "[REDACTED]"


@pytest.mark.django_db
class TestModelIntegration:
    """Test model integration and complex scenarios."""

    @pytest.mark.parametrize(
        "langchain_type,template_data",
        [
            ("PromptTemplate", {"template": "Hello {name}", "input_variables": ["name"]}),
            (
                "ChatPromptTemplate",
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You're an assistant who's good at {ability}",
                        },
                        {"MessagesPlaceholder": "history"},
                        {"role": "human", "content": "{question}"},
                    ],
                    "input_variables": ["ability", "history", "question"],
                },
            ),
        ],
    )
    def test_prompt_to_langchain_conversion(self, langchain_type, template_data):
        """Test conversion of prompts to LangChain objects."""
        template_data.update(
            {
                "langchain_type": langchain_type,
                "input_variables": template_data.get("input_variables", []),
            }
        )

        prompt = Prompt.objects.create(name=f"{langchain_type}-test", prompt_template=template_data)

        langchain_prompt = prompt.to_langchain_prompt()
        assert langchain_prompt is not None

    def test_workflow_to_langchain_chain(self):
        """Test workflow to LangChain chain conversion."""
        prompt = baker.make(
            Prompt,
            name="TestPrompt",
            is_active=True,
            prompt_template={
                "langchain_type": "PromptTemplate",
                "template": "Hello {name}",
                "input_variables": ["name"],
            },
        )

        workflow = baker.make(
            Workflow, workflow_definition=[{"type": "prompt", "name": prompt.name}]
        )

        chain = workflow.to_langchain_chain()
        assert chain is not None

        chain_with_logging = workflow.to_langchain_chain(log="true")
        assert chain_with_logging is not None

    def test_model_managers(self):
        """Test custom model managers."""
        active_session = baker.make(ChatSession, is_active=True)
        archived_session = baker.make(ChatSession, is_active=False)

        active_message = baker.make(ChatHistory, session=active_session, is_deleted=False)
        deleted_message = baker.make(ChatHistory, session=active_session, is_deleted=True)

        assert active_session in ChatSession.objects.active()
        assert archived_session in ChatSession.objects.archived()
        assert archived_session not in ChatSession.objects.active()

        assert active_message in ChatHistory.objects.active()
        assert deleted_message in ChatHistory.objects.deleted()
        assert deleted_message not in ChatHistory.objects.active()

        session_messages = ChatHistory.objects.for_session(active_session)
        assert active_message in session_messages
        assert deleted_message not in session_messages

        all_session_messages = ChatHistory.objects.all_for_session(active_session)
        assert active_message in all_session_messages
        assert deleted_message in all_session_messages
