"""
Streamlined integration tests for vanilla Django project API endpoints.

This module tests the API functionality with minimal redundancy using pytest parameterization.
"""

import json
import uuid
from unittest.mock import patch, MagicMock

import pytest
from django.test import Client
from django.urls import reverse
from django.contrib.auth import get_user_model

from django_chain.models import Prompt, Workflow, ChatSession, ChatHistory, InteractionLog
from examples.vanilla_django.example.models import ExampleDocument

User = get_user_model()


@pytest.fixture
def api_client():
    """API client with authenticated user."""
    client = Client()
    user = User.objects.create_user(
        username="apiuser", email="api@example.com", password="apipass123"
    )
    client.force_login(user)
    return client, user


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    with patch("django_chain.utils.llm_client.create_llm_chat_client") as mock:
        mock_response = MagicMock()
        mock_response.content = "Mocked LLM response"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock.return_value = mock_llm
        yield mock


@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing."""
    prompt = Prompt.objects.create(
        name="test_prompt",
        prompt_template={
            "langchain_type": "PromptTemplate",
            "template": "Test: {input}",
            "input_variables": ["input"],
        },
        input_variables=["input"],
        is_active=True,
    )
    return Workflow.objects.create(
        name="test_workflow",
        description="Test workflow",
        prompt=prompt,
        workflow_definition=[
            {"type": "prompt", "name": "test_prompt"},
            {"type": "llm", "provider": "fake"},
        ],
        is_active=True,
    )


@pytest.mark.skip()
@pytest.mark.django_db
class TestAPIEndpoints:
    """Consolidated API endpoint tests using parameterization."""

    @pytest.mark.parametrize(
        "endpoint,expected_content",
        [
            ("example:dashboard", "Django-Chain Dashboard"),
            ("example:api_overview", "django_chain_features"),
            ("example:prompt_examples", "prompts"),
            ("example:workflow_examples", "workflows"),
            ("example:chat_demo", "recent_sessions"),
            ("example:vector_demo", "documents"),
            ("example:llm_test", "providers"),
            ("example:interaction_logs", "logs"),
            ("example:custom_workflow", "html"),
        ],
    )
    def test_get_endpoints_return_expected_content(self, api_client, endpoint, expected_content):
        """Test that GET endpoints return expected content."""
        client, _ = api_client
        response = client.get(reverse(endpoint))
        assert response.status_code == 200
        if expected_content in ["django_chain_features"]:
            # JSON response
            data = response.json()
            assert expected_content in data
        else:
            # HTML response
            assert expected_content.encode() in response.content

    @pytest.mark.parametrize(
        "prompt_type,expected_langchain_type",
        [
            ("simple", "PromptTemplate"),
            ("chat", "ChatPromptTemplate"),
        ],
    )
    def test_prompt_creation_types(self, api_client, prompt_type, expected_langchain_type):
        """Test different prompt creation types."""
        client, _ = api_client
        data = {"example_type": prompt_type}
        response = client.post(
            reverse("example:prompt_examples"), json.dumps(data), content_type="application/json"
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "prompt_id" in result

        # Verify database
        prompt = Prompt.objects.get(id=result["prompt_id"])
        assert prompt.prompt_template["langchain_type"] == expected_langchain_type

    @pytest.mark.parametrize(
        "workflow_type,expected_steps",
        [
            ("simple", 2),  # prompt + llm
            ("with_parser", 3),  # prompt + llm + parser
        ],
    )
    def test_workflow_creation_types(self, api_client, workflow_type, expected_steps):
        """Test different workflow creation types."""
        client, _ = api_client
        data = {"workflow_type": workflow_type}
        response = client.post(
            reverse("example:workflow_examples"), json.dumps(data), content_type="application/json"
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"

        # Verify workflow structure
        workflow = Workflow.objects.get(id=result["workflow_id"])
        assert len(workflow.workflow_definition) == expected_steps

    def test_workflow_execution(self, api_client, sample_workflow, mock_llm_client):
        """Test workflow execution with session management."""
        client, user = api_client
        session_uuid = str(uuid.uuid4())

        data = {
            "workflow_name": sample_workflow.name,
            "input": {"input": "test input"},
            "session_id": session_uuid,
        }
        response = client.post(
            reverse("example:execute_workflow"), json.dumps(data), content_type="application/json"
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert result["session_id"] == session_uuid

        # Verify interaction log was created
        assert InteractionLog.objects.filter(workflow=sample_workflow).exists()

    def test_chat_session_management(self, api_client, mock_llm_client):
        """Test chat session creation and message handling."""
        client, user = api_client

        # Test new session creation
        data = {"message": "Hello"}
        response = client.post(
            reverse("example:chat_demo"), json.dumps(data), content_type="application/json"
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "session_id" in result

        session_id = result["session_id"]
        session = ChatSession.objects.get(session_id=session_id)
        assert session.messages.count() == 2  # user + assistant

        # Test existing session continuation
        data = {"message": "Continue", "session_id": session_id}
        response = client.post(
            reverse("example:chat_demo"), json.dumps(data), content_type="application/json"
        )

        assert response.status_code == 200
        session.refresh_from_db()
        assert session.messages.count() == 4  # 2 previous + 2 new

    @pytest.mark.parametrize(
        "operation,required_field,error_message",
        [
            ("add_document", "content", "Content required"),
            ("search", "query", "Query required"),
        ],
    )
    def test_vector_store_error_handling(
        self, api_client, operation, required_field, error_message
    ):
        """Test vector store operation error handling."""
        client, _ = api_client
        data = {"operation": operation}  # Missing required field

        response = client.post(
            reverse("example:vector_demo"), json.dumps(data), content_type="application/json"
        )

        assert response.status_code == 400
        result = response.json()
        assert result["status"] == "error"
        assert result["message"] == error_message

    @patch("django_chain.services.vector_store_manager.VectorStoreManager.add_documents_to_store")
    def test_vector_store_document_operations(self, mock_add_docs, api_client):
        """Test vector store document add and search operations."""
        client, _ = api_client
        mock_add_docs.return_value = None

        # Test document addition
        data = {
            "operation": "add_document",
            "title": "Test Doc",
            "content": "Test content",
        }
        response = client.post(
            reverse("example:vector_demo"), json.dumps(data), content_type="application/json"
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert ExampleDocument.objects.filter(title="Test Doc").exists()

    @pytest.mark.parametrize(
        "test_type,provider",
        [
            ("chat", "fake"),
            ("embedding", "fake"),
        ],
    )
    def test_llm_testing_endpoints(self, api_client, test_type, provider):
        """Test LLM testing functionality."""
        client, _ = api_client

        with patch(f"django_chain.utils.llm_client.create_llm_{test_type}_client") as mock_client:
            if test_type == "chat":
                mock_response = MagicMock()
                mock_response.content = "Test response"
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = mock_response
                mock_client.return_value = mock_llm
            else:  # embedding
                mock_client.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3]]

            data = {
                "test_type": test_type,
                "provider": provider,
                "message": "test",
            }
            response = client.post(
                reverse("example:llm_test"), json.dumps(data), content_type="application/json"
            )

            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "success"
            assert result["provider"] == provider

    @pytest.mark.parametrize(
        "error_type,expected_status",
        [
            ("prompt_validation", 400),
            ("workflow_validation", 400),
            ("generic", 500),
        ],
    )
    def test_error_handling_demo(self, api_client, error_type, expected_status):
        """Test error handling demonstrations."""
        client, _ = api_client
        data = {"error_type": error_type}

        response = client.post(
            reverse("example:error_demo"), json.dumps(data), content_type="application/json"
        )

        assert response.status_code == expected_status
        result = response.json()
        assert "status" in result
        assert "error" in result["status"] or "validation" in result["status"]

    def test_custom_workflow_builder(self, api_client, mock_llm_client):
        """Test custom workflow building functionality."""
        client, _ = api_client

        response = client.post(reverse("example:custom_workflow"), content_type="application/json")

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "workflow_id" in result
        assert result["steps_executed"] == 3

    @pytest.mark.parametrize(
        "invalid_data,expected_error",
        [
            ("invalid json", "Unexpected error"),  # JSON decode error
            ({"missing": "required_field"}, "required"),  # Missing required fields
        ],
    )
    def test_api_error_handling(self, api_client, invalid_data, expected_error):
        """Test API error handling with various invalid inputs."""
        client, _ = api_client

        if isinstance(invalid_data, str):
            response = client.post(
                reverse("example:prompt_examples"), invalid_data, content_type="application/json"
            )
        else:
            response = client.post(
                reverse("example:chat_demo"),
                json.dumps(invalid_data),
                content_type="application/json",
            )

        assert response.status_code in [400, 500]
        result = response.json()
        assert result["status"] == "error"

    def test_unicode_handling(self, api_client):
        """Test proper Unicode handling in API endpoints."""
        client, _ = api_client
        data = {
            "prompt_data": {
                "name": "unicode_test_🚀",
                "template": {
                    "langchain_type": "PromptTemplate",
                    "template": "Unicode: {input} 🌟",
                    "input_variables": ["input"],
                },
                "input_variables": ["input"],
            }
        }

        response = client.post(
            reverse("example:prompt_examples"),
            json.dumps(data, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"

        # Verify Unicode preservation
        prompt = Prompt.objects.get(name="unicode_test_🚀")
        assert "🌟" in prompt.prompt_template["template"]


@pytest.mark.skip()
@pytest.mark.django_db
class TestAPIConsistency:
    """Test API response format consistency."""

    @pytest.mark.parametrize(
        "endpoint,data",
        [
            ("example:prompt_examples", {"example_type": "simple"}),
            ("example:workflow_examples", {"workflow_type": "simple"}),
        ],
    )
    def test_success_response_format_consistency(self, api_client, endpoint, data):
        """Test that success responses follow consistent format."""
        client, _ = api_client
        response = client.post(reverse(endpoint), json.dumps(data), content_type="application/json")

        result = response.json()
        assert "status" in result
        assert result["status"] == "success"

    @pytest.mark.parametrize(
        "endpoint,invalid_data",
        [
            ("example:chat_demo", {}),  # Missing message
            ("example:execute_workflow", {"workflow_name": "nonexistent"}),  # Missing workflow
        ],
    )
    def test_error_response_format_consistency(self, api_client, endpoint, invalid_data):
        """Test that error responses follow consistent format."""
        client, _ = api_client
        response = client.post(
            reverse(endpoint), json.dumps(invalid_data), content_type="application/json"
        )

        result = response.json()
        assert "status" in result
        assert result["status"] == "error"
        assert "message" in result
