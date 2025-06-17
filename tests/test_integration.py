"""
Integration tests for django-chain using a test project.
"""

import json

import pytest
from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from django_chain.models import LLMChain
from django_chain.models import ChatSession
from django_chain.models import LLMInteractionLog
from django_chain.services.llm_client import LLMClient
from django_chain.services.vector_store_manager import VectorStoreManager
from tests.test_project.testapp.models import TestChain, TestDocument


class TestProjectIntegration(TestCase):
    """Integration tests using the test project."""

    def setUp(self):
        """Set up test environment."""
        self.client = Client()
        self.user = get_user_model().objects.create_user(
            username="testuser", password="testpass123"
        )
        self.client.login(username="testuser", password="testpass123")

        # Get model classes
        self.LLMChain = LLMChain
        self.ChatSession = ChatSession
        self.LLMInteractionLog = LLMInteractionLog

    @pytest.mark.skip()
    def test_llm_call(self):
        """Test LLM call through the test project."""
        response = self.client.post(reverse("testapp:test_llm"), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["status"], "success")
        self.assertIn("response", data)

    @pytest.mark.skip()
    def test_chain_execution(self):
        """Test chain execution through the test project."""
        response = self.client.post(reverse("testapp:test_chain"), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["status"], "success")
        self.assertIn("response", data)

        # Verify chain was created
        self.assertTrue(self.LLMChain.objects.filter(name="test_chain").exists())

    def test_chat_session(self):
        """Test chat session through the test project."""
        response = self.client.post(reverse("testapp:test_chat"), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["status"], "success")
        self.assertIn("session_id", data)
        self.assertEqual(data["message_count"], 1)

        # Verify session and message were created
        session = self.ChatSession.objects.get(id=data["session_id"])
        self.assertEqual(session.messages.count(), 1)

    @pytest.mark.skip()
    def test_vector_store(self):
        """Test vector store operations through the test project."""
        response = self.client.post(
            reverse("testapp:test_vector_store"), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["status"], "success")
        self.assertIn("document_id", data)
        self.assertIn("search_results", data)

        # Verify document was created
        self.assertTrue(TestDocument.objects.filter(id=data["document_id"]).exists())

    @pytest.mark.skip()
    def test_llm_interaction_logging(self):
        """Test that LLM interactions are properly logged."""
        # Make an LLM call
        self.client.post(reverse("testapp:test_llm"), content_type="application/json")

        # Verify interaction was logged
        self.assertTrue(self.LLMInteractionLog.objects.exists())
        log = self.LLMInteractionLog.objects.first()
        self.assertEqual(log.status, "success")
        self.assertEqual(log.provider, "fake")

    @pytest.mark.skip()
    def test_model_relationships(self):
        """Test relationships between django-chain and test app models."""
        # Create a chain
        chain = self.LLMChain.objects.create(
            name="test_chain",
            prompt_template="Hello, {name}!",
            model_name="fake-model",
            input_variables=["name"],
        )

        # Create a test chain
        test_chain = TestChain.objects.create(name="test_chain_wrapper", chain=chain)

        # Create a chat session
        session = self.ChatSession.objects.create(
            title="Test Chat", llm_config={"model_name": "fake-model"}
        )

        # Create a test chat
        test_chat = TestChat.objects.create(name="test_chat_wrapper", session=session)

        # Verify relationships
        self.assertEqual(test_chain.chain, chain)


@pytest.mark.skip()
@pytest.mark.django_db()
def test_chat_integration() -> None:
    """Test chat integration."""
    client = Client()
    response = client.post(
        reverse("chat"),
        {"message": "Hello", "session_id": "test-session"},
        content_type="application/json",
    )
    assert response.status_code == 200
    assert "response" in response.json()


@pytest.mark.skip()
@pytest.mark.django_db()
def test_vector_search_integration() -> None:
    """Test vector search integration."""
    client = Client()
    response = client.post(
        reverse("vector_search"),
        {"query": "test query", "k": 5},
        content_type="application/json",
    )
    assert response.status_code == 200
    assert "results" in response.json()


@pytest.mark.skip()
@pytest.mark.django_db()
def test_chain_execution_integration() -> None:
    """Test chain execution integration."""
    client = LLMClient()
    response = client.execute_chain(
        chain_name="test-chain", inputs={"var1": "value1", "var2": "value2"}
    )
    assert response is not None


@pytest.mark.skip()
@pytest.mark.django_db()
def test_vector_store_integration() -> None:
    """Test vector store integration."""
    manager = VectorStoreManager()

    # Add test documents
    documents = [
        {"content": "Test document 1", "metadata": {"source": "test1"}},
        {"content": "Test document 2", "metadata": {"source": "test2"}},
    ]
    manager.add_documents(documents)

    # Test retrieval
    results = manager.retrieve_documents("Test document", k=2)
    assert len(results) == 2
    assert results[0]["page_content"] == "Test document 1"
    assert results[1]["page_content"] == "Test document 2"


@pytest.mark.skip()
@pytest.mark.django_db()
def test_chat_session_integration() -> None:
    """Test chat session integration."""
    client = LLMClient()

    # Create a new session
    session_id = "test-session"
    response1 = client.chat("Hello", session_id)
    assert response1 is not None

    # Continue the conversation
    response2 = client.chat("How are you?", session_id)
    assert response2 is not None

    # Verify session history
    history = client.get_chat_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"


@pytest.mark.skip()
@pytest.mark.django_db()
def test_error_handling_integration() -> None:
    """Test error handling integration."""
    client = Client()

    # Test invalid chat request
    response = client.post(
        reverse("chat"), {"invalid_field": "value"}, content_type="application/json"
    )
    assert response.status_code == 400
    assert "error" in response.json()

    # Test invalid vector search request
    response = client.post(
        reverse("vector_search"),
        {"invalid_field": "value"},
        content_type="application/json",
    )
    assert response.status_code == 400


@pytest.mark.skip()
@pytest.mark.django_db()
def test_performance_integration() -> None:
    """Test performance integration."""
    manager = VectorStoreManager()

    # Add multiple documents
    documents = [
        {"content": f"Test document {i}", "metadata": {"source": f"test{i}"}} for i in range(100)
    ]
    manager.add_documents(documents)

    # Test retrieval performance
    results = manager.retrieve_documents("Test document", k=10)
    assert len(results) == 10
