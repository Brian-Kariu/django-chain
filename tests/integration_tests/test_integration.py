"""
Streamlined integration tests for django-chain components.

These tests verify that components work together correctly in a Django environment.
"""

import json
import uuid
from unittest.mock import patch, MagicMock

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

from django_chain.models import Prompt, Workflow, ChatSession, ChatHistory, InteractionLog
from django_chain.services.vector_store_manager import VectorStoreManager
from examples.vanilla_django.example.models import ExampleDocument, ExampleWorkflowExecution

User = get_user_model()


@pytest.fixture
def auth_client():
    """Authenticated client with user."""
    client = Client()
    user = User.objects.create_user(
        username="testuser", email="test@example.com", password="testpass123"
    )
    client.force_login(user)
    return client, user


@pytest.fixture
def sample_data():
    """Sample test data."""
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

    workflow = Workflow.objects.create(
        name="test_workflow",
        description="Test workflow",
        prompt=prompt,
        workflow_definition=[
            {"type": "prompt", "name": "test_prompt"},
            {"type": "llm", "provider": "fake"},
        ],
        is_active=True,
    )

    return {"prompt": prompt, "workflow": workflow}


@pytest.mark.skip()
@pytest.mark.django_db
class TestIntegration:
    """Consolidated integration tests."""

    def test_project_navigation(self, auth_client):
        """Test main project endpoints are accessible."""
        client, _ = auth_client

        endpoints = ["/", reverse("example:dashboard"), reverse("example:api_overview")]
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200

    @pytest.mark.parametrize(
        "workflow_type,expected_steps",
        [
            ("simple", 2),
            ("with_parser", 3),
        ],
    )
    def test_workflow_creation_and_execution(self, auth_client, workflow_type, expected_steps):
        """Test workflow creation and execution flow."""
        client, user = auth_client

        # Create workflow
        data = {"workflow_type": workflow_type}
        response = client.post(
            reverse("example:workflow_examples"), json.dumps(data), content_type="application/json"
        )
        assert response.status_code == 200

        result = response.json()
        workflow = Workflow.objects.get(id=result["workflow_id"])
        assert len(workflow.workflow_definition) == expected_steps

        # Execute workflow
        with patch("django_chain.utils.llm_client.create_llm_chat_client") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_llm.return_value.invoke.return_value = mock_response

            exec_data = {
                "workflow_name": workflow.name,
                "input": {"input": "test"},
            }
            exec_response = client.post(
                reverse("example:execute_workflow"),
                json.dumps(exec_data),
                content_type="application/json",
            )
            assert exec_response.status_code == 200

    def test_chat_session_workflow(self, auth_client):
        """Test complete chat session workflow."""
        client, user = auth_client

        with patch("django_chain.utils.llm_client.create_llm_chat_client") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "Chat response"
            mock_llm.return_value.invoke.return_value = mock_response

            # Start new chat
            data = {"message": "Hello"}
            response = client.post(
                reverse("example:chat_demo"), json.dumps(data), content_type="application/json"
            )

            assert response.status_code == 200
            result = response.json()
            session_id = result["session_id"]

            # Verify session created
            session = ChatSession.objects.get(session_id=session_id)
            assert session.messages.count() == 2  # user + assistant

            # Continue conversation
            continue_data = {"message": "Continue", "session_id": session_id}
            continue_response = client.post(
                reverse("example:chat_demo"),
                json.dumps(continue_data),
                content_type="application/json",
            )

            assert continue_response.status_code == 200
            session.refresh_from_db()
            assert session.messages.count() == 4

    @patch("django_chain.services.vector_store_manager.VectorStoreManager.add_documents_to_store")
    def test_vector_store_integration(self, mock_add_docs, auth_client):
        """Test vector store document management."""
        client, user = auth_client
        mock_add_docs.return_value = None

        # Add document
        data = {
            "operation": "add_document",
            "title": "Test Document",
            "content": "Test content for vector store",
        }
        response = client.post(
            reverse("example:vector_demo"), json.dumps(data), content_type="application/json"
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"

        # Verify document created
        doc = ExampleDocument.objects.get(id=result["document_id"])
        assert doc.title == "Test Document"

    def test_example_models_integration(self, auth_client, sample_data):
        """Test integration with example models."""
        client, user = auth_client
        workflow = sample_data["workflow"]

        # Test ExampleDocument
        doc = ExampleDocument.objects.create(
            title="Integration Test Doc",
            content="Test content",
            author=user,
            tags=["test"],
        )
        assert str(doc) == "Integration Test Doc"
        assert not doc.is_indexed

        doc.mark_as_indexed()
        assert doc.is_indexed

        # Test ExampleWorkflowExecution
        execution = ExampleWorkflowExecution.objects.create(
            workflow=workflow,
            user=user,
            input_data={"test": "input"},
            output_data={"test": "output"},
            execution_time_ms=100,
            success=True,
        )
        assert "Success" in str(execution)

    @patch("django_chain.utils.llm_client.create_llm_chat_client")
    def test_end_to_end_workflow_scenario(self, mock_llm, auth_client):
        """Test complete end-to-end workflow scenario."""
        client, user = auth_client

        # Mock LLM
        mock_response = MagicMock()
        mock_response.content = "End-to-end response"
        mock_llm.return_value.invoke.return_value = mock_response

        # 1. Create prompt
        prompt_data = {"example_type": "simple"}
        prompt_response = client.post(
            reverse("example:prompt_examples"),
            json.dumps(prompt_data),
            content_type="application/json",
        )
        assert prompt_response.status_code == 200

        # 2. Create workflow
        workflow_data = {"workflow_type": "simple"}
        workflow_response = client.post(
            reverse("example:workflow_examples"),
            json.dumps(workflow_data),
            content_type="application/json",
        )
        assert workflow_response.status_code == 200
        workflow_result = workflow_response.json()

        # 3. Execute workflow with chat session
        session_id = str(uuid.uuid4())
        exec_data = {
            "workflow_name": workflow_result["name"],
            "input": {"input": "test"},
            "session_id": session_id,
        }
        exec_response = client.post(
            reverse("example:execute_workflow"),
            json.dumps(exec_data),
            content_type="application/json",
        )

        assert exec_response.status_code == 200
        exec_result = exec_response.json()
        assert exec_result["session_id"] == session_id

        # 4. Verify all components created
        workflow = Workflow.objects.get(name=workflow_result["name"])
        session = ChatSession.objects.get(session_id=session_id)
        assert InteractionLog.objects.filter(workflow=workflow).exists()

    @pytest.mark.parametrize(
        "error_scenario,expected_status",
        [
            ("invalid_json", 500),
            ("missing_workflow", 404),
            ("invalid_operation", 400),
        ],
    )
    def test_error_handling_scenarios(self, auth_client, error_scenario, expected_status):
        """Test various error handling scenarios."""
        client, _ = auth_client

        if error_scenario == "invalid_json":
            response = client.post(
                reverse("example:prompt_examples"), "invalid json", content_type="application/json"
            )
        elif error_scenario == "missing_workflow":
            data = {"workflow_name": "nonexistent", "input": {}}
            response = client.post(
                reverse("example:execute_workflow"),
                json.dumps(data),
                content_type="application/json",
            )
        elif error_scenario == "invalid_operation":
            data = {"operation": "invalid"}
            response = client.post(
                reverse("example:vector_demo"), json.dumps(data), content_type="application/json"
            )

        assert response.status_code == expected_status
        if response.status_code != 404:  # 404 might not return JSON
            result = response.json()
            assert "error" in result.get("status", "").lower()

    def test_performance_monitoring(self, auth_client, sample_data):
        """Test that performance monitoring works correctly."""
        client, user = auth_client
        workflow = sample_data["workflow"]

        # Execute workflow multiple times to generate logs
        with patch("django_chain.utils.llm_client.create_llm_chat_client") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "Performance test"
            mock_llm.return_value.invoke.return_value = mock_response

            for i in range(3):
                data = {
                    "workflow_name": workflow.name,
                    "input": {"input": f"test {i}"},
                }
                response = client.post(
                    reverse("example:execute_workflow"),
                    json.dumps(data),
                    content_type="application/json",
                )
                assert response.status_code == 200

        # Verify interaction logs created
        logs = InteractionLog.objects.filter(workflow=workflow)
        assert logs.count() == 3

        # Test interaction logs view
        logs_response = client.get(reverse("example:interaction_logs"))
        assert logs_response.status_code == 200
