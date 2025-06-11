import pytest
from django.test import TestCase

from django_chain.models.chains import LLMChain
from django_chain.models.chat import ChatMessage, ChatSession


class TestLLMChain(TestCase):
    def setUp(self):
        """Set up test environment."""
        self.LLMChain = get_llm_chain_model()

    def test_create_chain(self):
        """Test creating a new LLMChain instance."""
        chain = self.LLMChain.objects.create(
            name="test_chain",
            prompt_template="Test prompt: {input}",
            model_name="gpt-3.5-turbo",
        )
        self.assertEqual(chain.name, "test_chain")
        self.assertEqual(chain.prompt_template, "Test prompt: {input}")
        self.assertEqual(chain.model_name, "gpt-3.5-turbo")
        self.assertEqual(chain.temperature, 0.7)  # default value

    def test_format_prompt(self):
        """Test formatting a prompt with context."""
        chain = self.LLMChain.objects.create(
            name="test_chain",
            prompt_template="Test prompt: {input}",
            model_name="gpt-3.5-turbo",
        )
        formatted = chain.format_prompt({"input": "test input"})
        self.assertEqual(formatted, "Test prompt: test input")

    def test_get_chain_not_implemented(self):
        """Test that get_chain raises NotImplementedError."""
        chain = self.LLMChain.objects.create(
            name="test_chain",
            prompt_template="Test prompt: {input}",
            model_name="gpt-3.5-turbo",
        )
        with pytest.raises(NotImplementedError):
            chain.get_chain()


@pytest.mark.django_db()
def test_chain_model() -> None:
    """Test Chain model."""
    chain = LLMChain.objects.create(
        name="Test Chain",
        prompt_template="Test template",
        model_name="gpt-3.5-turbo",
        provider="openai",
        temperature=0.7,
        max_tokens=100,
        input_variables=["var1", "var2"],
        output_parser={"type": "str"},
        is_active=True,
    )
    assert chain.name == "Test Chain"
    assert chain.prompt_template == "Test template"
    assert chain.model_name == "gpt-3.5-turbo"
    assert chain.provider == "openai"
    assert chain.temperature == 0.7
    assert chain.max_tokens == 100
    assert chain.input_variables == ["var1", "var2"]
    assert chain.output_parser == {"type": "str"}
    assert chain.is_active is True


@pytest.mark.django_db()
def test_chat_session_model() -> None:
    """Test ChatSession model."""
    session = ChatSession.objects.create(
        session_id="test-session",
        title="Test Session",
        llm_config={"model": "gpt-3.5-turbo", "temperature": 0.7},
    )
    assert session.session_id == "test-session"
    assert session.title == "Test Session"
    assert session.llm_config["model"] == "gpt-3.5-turbo"
    assert session.llm_config["temperature"] == 0.7


@pytest.mark.django_db()
def test_chat_message_model() -> None:
    """Test ChatMessage model."""
    session = ChatSession.objects.create(session_id="test-session", title="Test Session")
    message = ChatMessage.objects.create(
        session=session, content="Hello, world!", role="user", token_count=5, order=1
    )
    assert message.session == session
    assert message.content == "Hello, world!"
    assert message.role == "user"
    assert message.token_count == 5
    assert message.order == 1
