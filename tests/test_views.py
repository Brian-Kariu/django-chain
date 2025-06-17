import json
from django.core import serializers
from model_bakery import baker
import pytest
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from django_chain.models import (
    MessageTypes,
    Prompt,
    PromptTemplate,
    PromptTemplateRendering,
    PromptTemplateTypes,
    Message,
)
from django.urls import reverse

from tests.test_project.testapp.models import TestPrompt


@override_settings(ROOT_URLCONF="test_project.testapp.urls")
@pytest.mark.django_db
class TestPromptView(TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_get_prompt(self):
        prompt_1 = baker.make(
            TestPrompt,
            name="test_prompt_1",
            input_variables=["msg"],
            optional_variables=["placeholder"],
        )
        prompt_2 = baker.make(
            TestPrompt,
            name="test_prompt_2",
            input_variables=["msg"],
            optional_variables=["placeholder"],
        )

        response = self.client.get(reverse("test_prompt_by_id", kwargs={"prompt_id": prompt_1.id}))
        response_data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(len(response_data) == 1, "The list of prompts should be one.")

        response = self.client.get(reverse("test_prompt", kwargs={}))
        response_data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(len(response_data) == 2, "The list of prompts should two.")


@pytest.mark.skip()
class TestLLMView(TestCase):
    def setUp(self):
        self.client = Client()
        self.LLMChain = get_llm_chain_model()
        self.chain = self.LLMChain.objects.create(
            name="test_chain",
            prompt_template="Test prompt: {input}",
            model_name="gpt-3.5-turbo",
        )

    def test_view_without_chain(self):
        """Test view behavior when no chain is configured."""
        response = self.client.get(reverse("test_llm"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "No chain configured"})

    def test_view_with_input(self):
        """Test view behavior with input parameter."""
        # Note: This test will fail until we implement the actual chain
        # functionality. It's here as a placeholder for future implementation.


@pytest.mark.skip()
@pytest.mark.django_db()
def test_chat_view() -> None:
    """Test chat view."""
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
def test_vector_search_view() -> None:
    """Test vector search view."""
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
def test_document_view() -> None:
    """Test document view."""
    client = Client()
    response = client.post(
        reverse("document"),
        {"title": "Test Document", "content": "Test content"},
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["title"] == "Test Document"
    assert data["content"] == "Test content"
