import json
import uuid
from unittest.mock import Mock, patch, MagicMock

import pytest
from django.test import RequestFactory
from django.core.exceptions import ValidationError
from model_bakery import baker

from django_chain.models import InteractionLog, Prompt, Workflow
from django_chain.views import (
    InteractionLogDetailView,
    InteractionLogListCreateView,
    PromptListCreateView,
    PromptDetailView,
    PromptActivateDeactivateView,
    WorkflowListCreateView,
    WorkflowDetailView,
    WorkflowActivateDeactivateView,
    ExecuteWorkflowView,
    chat_view,
    vector_search_view,
    serialize_queryset,
)


@pytest.fixture
def request_factory():
    return RequestFactory()


@pytest.fixture
def sample_prompt():
    return baker.make(
        Prompt,
        name="SamplePrompt",
        prompt_template={
            "langchain_type": "PromptTemplate",
            "template": "Hello {name}",
            "input_variables": ["name"],
        },
        input_variables=["name"],
        is_active=True,
    )


@pytest.fixture
def sample_workflow():
    return baker.make(
        Workflow,
        name="SampleWorkflow",
        workflow_definition=[
            {"type": "prompt", "name": "SimpleGreetingPrompt"},
            {"type": "llm", "config": {"temperature": 0.4}},
        ],
        is_active=True,
    )


@pytest.mark.django_db
class TestPromptViews:
    """Test prompt-related views."""

    def test_prompt_list_get(self, request_factory):
        """Test GET request to prompt list."""
        request = request_factory.get("/prompts/")
        view = PromptListCreateView()
        response = view.get(request)
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "include_inactive,expected_filter",
        [
            ("true", True),
            ("false", False),
            (None, False),
        ],
    )
    def test_prompt_list_filtering(self, request_factory, include_inactive, expected_filter):
        """Test prompt list filtering by active status."""
        baker.make(
            Prompt,
            prompt_template={
                "langchain_type": "ChatPromptTemplate",
                "messages": [
                    {"role": "system", "content": "You're an assistant who's good at {ability}"},
                    {"MessagesPlaceholder": "history"},
                    {"role": "human", "content": "{question}"},
                ],
            },
            is_active=True,
        )
        baker.make(
            Prompt,
            prompt_template={
                "langchain_type": "ChatPromptTemplate",
                "messages": [
                    {"role": "system", "content": "You're an assistant who's good at {ability}"},
                    {"MessagesPlaceholder": "history"},
                    {"role": "human", "content": "{question}"},
                ],
            },
            is_active=False,
        )

        url = "/prompts/"
        if include_inactive:
            url += f"?include_inactive={include_inactive}"

        request = request_factory.get(url)
        view = PromptListCreateView()
        response = view.get(request)

        assert response.status_code == 200

    @pytest.mark.skip()
    @patch("django_chain.models.Prompt.create_new_version")
    def test_prompt_create_success(self, mock_create, request_factory):
        """Test successful prompt creation."""
        mock_prompt = Mock()
        mock_prompt.to_dict.return_value = {"id": str(uuid.uuid4()), "name": "test"}
        mock_create.return_value = mock_prompt

        request_data = {
            "name": "test_prompt",
            "prompt_template": {
                "langchain_type": "ChatPromptTemplate",
                "messages": [
                    {"role": "system", "content": "You're an assistant who's good at {ability}"},
                    {"MessagesPlaceholder": "history"},
                    {"role": "human", "content": "{question}"},
                ],
            },
            "input_variables": ["ability", "history", "question"],
        }

        request = request_factory.post("/prompts/", data=json.dumps(request_data))
        request.json_body = request_data

        view = PromptListCreateView()
        response = view.post(request)

        assert response.status_code == 201
        mock_create.assert_called_once()

    @pytest.mark.parametrize(
        "action,method",
        [
            ("activate", "activate"),
            ("deactivate", "deactivate"),
        ],
    )
    def test_prompt_activate_deactivate(self, request_factory, sample_prompt, action, method):
        """Test prompt activation/deactivation."""
        request = request_factory.post(f"/prompts/{sample_prompt.id}/{action}/")

        with patch.object(sample_prompt, method) as mock_method:
            view = PromptActivateDeactivateView()
            view.get_object = Mock(return_value=sample_prompt)
            response = view.post(request, pk=str(sample_prompt.id), action=action)

        assert response.status_code == 200
        mock_method.assert_called_once()

    @pytest.mark.parametrize(
        "http_method,view_method",
        [
            ("get", "get"),
            ("put", "put"),
            ("delete", "delete"),
        ],
    )
    def test_prompt_detail_methods(self, request_factory, sample_prompt, http_method, view_method):
        """Test prompt detail view methods."""
        request = getattr(request_factory, http_method)(f"/prompts/{sample_prompt.id}/")

        if http_method == "put":
            request.json_body = {"input_variables": ["updated"]}

        view = PromptDetailView()
        response = getattr(view, view_method)(request, pk=str(sample_prompt.id))

        if http_method == "delete":
            assert response.status_code == 204
        else:
            assert response.status_code == 200


@pytest.mark.django_db
class TestWorkflowViews:
    """Test workflow-related views."""

    def test_workflow_list_get(self, request_factory):
        """Test GET request to workflow list."""
        request = request_factory.get("/workflows/")
        view = WorkflowListCreateView()
        response = view.get(request)
        assert response.status_code == 200

    @pytest.mark.skip()
    @patch("django_chain.models.Workflow.create_new_version")
    def test_workflow_create_success(self, mock_create, request_factory, sample_prompt):
        """Test successful workflow creation."""
        mock_workflow = Mock()
        mock_workflow.to_dict.return_value = {"id": str(uuid.uuid4()), "name": "test"}
        mock_create.return_value = mock_workflow

        request_data = {
            "name": "test_workflow",
            "workflow_definition": [{"type": "prompt"}, {"type": "llm"}],
            "prompt": str(sample_prompt.id),
        }

        request = request_factory.post("/workflows/", data=json.dumps(request_data))
        request.json_body = request_data

        view = WorkflowListCreateView()
        response = view.post(request)

        assert response.status_code == 201
        mock_create.assert_called_once()

    @pytest.mark.skip()
    @pytest.mark.parametrize(
        "invalid_data,expected_status",
        [
            ({"name": "test", "workflow_definition": "invalid"}, 400),
            ({"workflow_definition": []}, 400),
        ],
    )
    def test_workflow_create_validation_errors(
        self, request_factory, invalid_data, expected_status
    ):
        """Test workflow creation validation errors."""
        request = request_factory.post("/workflows/", data=json.dumps(invalid_data))
        request.json_body = invalid_data

        view = WorkflowListCreateView()
        response = view.post(request)

        assert response.status_code == expected_status

    @pytest.mark.parametrize(
        "http_method,expected_status",
        [
            ("get", 200),
            # ("put", 200),
            ("delete", 204),
        ],
    )
    def test_workflow_detail_methods(
        self, request_factory, sample_workflow, http_method, expected_status
    ):
        """Test workflow detail view methods."""
        request = getattr(request_factory, http_method)(f"/workflows/{sample_workflow.id}/")

        if http_method == "put":
            request.json_body = {
                "description": "Updated description",
                "prompt": str(sample_workflow.prompt.id),
            }

        view = WorkflowDetailView()
        response = getattr(view, http_method)(request, pk=str(sample_workflow.id))

        assert response.status_code == expected_status


@pytest.mark.django_db
class TestExecuteWorkflowView:
    """Test workflow execution view."""

    @pytest.fixture
    def execution_view(self):
        return ExecuteWorkflowView()

    @pytest.fixture
    def test_workflow(self):
        prompt = baker.make(
            Prompt,
            name="TestPrompt",
            prompt_template={
                "langchain_type": "PromptTemplate",
                "template": "Test {input}",
                "input_variables": ["input"],
            },
            is_active=True,
        )
        return baker.make(
            Workflow,
            name="TestWorkflow",
            prompt=prompt,
            workflow_definition=[
                {"type": "prompt", "name": "TestPrompt"},
                {"type": "llm", "provider": "fake"},
            ],
            is_active=True,
        )

    @pytest.mark.skip()
    @pytest.mark.parametrize(
        "session_provided,creates_session",
        [
            (True, False),
            (False, True),
        ],
    )
    @patch("django_chain.views.getattr")
    @patch("django_chain.views._execute_and_log_workflow_step")
    def test_workflow_execution_with_session(
        self,
        mock_execute,
        mock_getattr,
        request_factory,
        execution_view,
        test_workflow,
        session_provided,
        creates_session,
    ):
        """Test workflow execution with session management."""
        mock_getattr.return_value = {}
        mock_execute.return_value = "Test response"

        session_id = str(uuid.uuid4()) if session_provided else None
        request_data = {
            "input": {"input": "test"},
            "session_id": session_id,
        }

        request = request_factory.post(
            f"/workflows/{test_workflow.name}/execute/",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        response = execution_view.post(request, name=test_workflow.name)

        assert response.status_code == 200
        result = json.loads(response.content)

        if session_provided:
            assert result["session_id"] == session_id
        else:
            assert "session_id" not in result

    @pytest.mark.parametrize(
        "error_scenario,expected_status",
        [
            ("nonexistent_workflow", 404),
            ("invalid_json", 400),
            ("invalid_session_id", 400),
        ],
    )
    def test_workflow_execution_errors(
        self, request_factory, execution_view, error_scenario, expected_status
    ):
        """Test workflow execution error scenarios."""
        if error_scenario == "nonexistent_workflow":
            request_data = {"input": {}}
            request = request_factory.post(
                "/workflows/nonexistent/execute/",
                data=json.dumps(request_data),
                content_type="application/json",
            )
            response = execution_view.post(request, name="nonexistent")

        elif error_scenario == "invalid_json":
            request = request_factory.post(
                "/workflows/test/execute/", data="invalid json", content_type="application/json"
            )
            response = execution_view.post(request, name="test")

        elif error_scenario == "invalid_session_id":
            request_data = {"input": {}, "session_id": "invalid-uuid"}
            request = request_factory.post(
                "/workflows/test/execute/",
                data=json.dumps(request_data),
                content_type="application/json",
            )
            response = execution_view.post(request, name="test")

        assert response.status_code == expected_status


@pytest.mark.django_db
class TestInteractionLogViews:
    """Test interaction log views."""

    def test_interaction_log_list(self, request_factory):
        """Test interaction log list view."""
        baker.make(InteractionLog, _quantity=3)

        request = request_factory.get("/logs/")
        view = InteractionLogListCreateView()
        response = view.get(request)

        assert response.status_code == 200

    def test_interaction_log_detail(self, request_factory):
        """Test interaction log detail view."""
        log = baker.make(InteractionLog)

        request = request_factory.get(f"/logs/{log.id}/")
        view = InteractionLogDetailView()
        response = view.get(request, pk=str(log.id))

        assert response.status_code == 200

    def test_interaction_log_delete(self, request_factory):
        """Test interaction log deletion."""
        log = baker.make(InteractionLog)

        request = request_factory.delete(f"/logs/{log.id}/")
        view = InteractionLogDetailView()
        response = view.delete(request, pk=str(log.id))

        assert response.status_code == 204


@pytest.mark.django_db
class TestUtilityViews:
    """Test utility views and functions."""

    @patch("django_chain.utils.llm_client.create_llm_chat_client")
    def test_chat_view_success(self, mock_client, request_factory):
        """Test successful chat view execution."""
        mock_response = Mock()
        mock_response.content = "Chat response"
        mock_client.return_value.invoke.return_value = mock_response

        data = {"message": "Hello", "session_id": str(uuid.uuid4())}
        request = request_factory.post(
            "/chat/", data=json.dumps(data), content_type="application/json"
        )

        response = chat_view(request)

        assert response.status_code == 200
        result = json.loads(response.content)
        assert "response" in result

    @pytest.mark.parametrize(
        "missing_field,error_message",
        [
            ("message", "Message is required"),
            ("query", "Query is required"),
        ],
    )
    def test_view_validation_errors(self, request_factory, missing_field, error_message):
        """Test validation errors in utility views."""
        if missing_field == "message":
            data = {"session_id": "test"}
            response = chat_view(
                request_factory.post("/chat/", json.dumps(data), content_type="application/json")
            )
        elif missing_field == "query":
            data = {"k": 5}
            response = vector_search_view(
                request_factory.post("/search/", json.dumps(data), content_type="application/json")
            )

        assert response.status_code == 400
        result = json.loads(response.content)
        assert error_message in result["error"]

    def test_serialize_queryset_utility(self):
        """Test serialize_queryset utility function."""
        prompts = baker.make(
            Prompt,
            prompt_template={
                "langchain_type": "ChatPromptTemplate",
                "messages": [
                    {"role": "system", "content": "You're an assistant who's good at {ability}"},
                    {"MessagesPlaceholder": "history"},
                    {"role": "human", "content": "{question}"},
                ],
            },
            _quantity=2,
        )
        for prompt in prompts:
            prompt.to_dict = Mock(return_value={"id": str(prompt.id)})

        result = serialize_queryset(prompts)
        assert len(result) == 2

        empty_result = serialize_queryset([])
        assert empty_result == []
