import json
import uuid
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase, RequestFactory, override_settings
from django.core.exceptions import ValidationError, ObjectDoesNotExist
from django.http import JsonResponse
from django.urls import reverse
import pytest
from model_bakery import baker
from faker import Faker

from django_chain.models import InteractionLog, Prompt, Workflow, UserInteraction
from django_chain.views import (
    InteractionLogDetailView,
    InteractionLogListCreateView,
    PromptListCreateView,
    PromptDetailView,
    PromptActivateDeactivateView,
    WorkflowListCreateView,
    WorkflowDetailView,
    WorkflowActivateDeactivateView,
    UserInteractionListView,
    UserInteractionDetailView,
    ExecuteWorkflowView,
    chat_view,
    vector_search_view,
    serialize_queryset,
    serialize_user_interaction,
)

fake = Faker()


@pytest.fixture
def request_factory():
    return RequestFactory()


@pytest.fixture
def sample_prompt():
    return baker.make(
        Prompt,
        name=fake.name(),
        prompt_template={
            "langchain_type": "ChatPromptTemplate",
            "messages": [
                {
                    "message_type": "system",
                    "template": "You are a helpful AI assistant for a tech company named 'TechSolutions'. Your goal is to provide concise and accurate support.",
                },
                {
                    "message_type": "human",
                    "template": "I have a problem with my {product_name}. The issue is: {issue_description}",
                },
                {
                    "message_type": "ai",
                    "template": "Thank you for contacting TechSolutions. I understand you're having an issue with your {product_name}. Can you provide more details?",
                },
                {
                    "message_type": "human",
                    "template": "Customer follow-up: {followup_detail}",
                },
            ],
            "input_variables": ["product_name", "issue_description", "followup_detail"],
        },
        input_variables=["var1", "var2"],
        is_active=True,
    )


@pytest.fixture
def sample_workflow():
    return baker.make(
        Workflow,
        name=fake.name(),
        description=fake.text(),
        workflow_definition=[
            {"type": "prompt", "name": "SimpleGreetingPrompt"},
            {"type": "llm", "config": {"temperature": 0.5}},
            {"type": "parser", "parser_type": "StrOutputParser"},
        ],
        is_active=True,
    )


@pytest.fixture
def sample_interactionlog():
    return baker.make(InteractionLog)


@pytest.fixture
def sample_user_interaction():
    workflow = baker.make(
        Workflow,
        name=fake.name(),
        workflow_definition=[
            {"type": "prompt", "name": "SimpleGreetingPrompt"},
            {"type": "llm", "config": {"temperature": 0.5}},
            {"type": "parser", "parser_type": "StrOutputParser"},
        ],
    )
    return baker.make(
        UserInteraction,
        workflow=workflow,
        session_id=uuid.uuid4(),
        user_identifier=fake.email(),
        input_data={"input": "test"},
        llm_output={"output": "response"},
        status="completed",
    )


class TestPromptListCreateView:
    @pytest.mark.django_db
    def test_get_list_prompts_success(self, request_factory, sample_prompt):
        request = request_factory.get("/prompts/")
        request.json_body = {}

        view = PromptListCreateView()
        response = view.get(request)

        assert response.status_code == 200
        data = json.loads(response.content)
        assert len(data) >= 1
        assert data[0]["name"] == sample_prompt.name

    @pytest.mark.django_db
    def test_get_list_prompts_with_filters(self, request_factory):
        active_prompt = baker.make(
            Prompt,
            name="active_prompt",
            prompt_template={"langchain_type": "ChatPromptTemplate"},
            is_active=True,
        )
        inactive_prompt = baker.make(
            Prompt,
            name="inactive_prompt",
            prompt_template={"langchain_type": "ChatPromptTemplate"},
            is_active=False,
        )

        # Test without include_inactive filter
        request = request_factory.get("/prompts/")
        request.json_body = {}

        view = PromptListCreateView()
        response = view.get(request)

        data = json.loads(response.content)
        assert len(data) == 1
        assert data[0]["name"] == active_prompt.name

    @pytest.mark.django_db
    def test_get_list_prompts_include_inactive(self, request_factory):
        baker.make(
            Prompt,
            name="active_prompt",
            prompt_template={"langchain_type": "ChatPromptTemplate"},
            is_active=True,
        )
        baker.make(
            Prompt,
            name="inactive_prompt",
            prompt_template={"langchain_type": "ChatPromptTemplate"},
            is_active=False,
        )

        request = request_factory.get("/prompts/?include_inactive=true")
        request.json_body = {}

        view = PromptListCreateView()
        response = view.get(request)

        data = json.loads(response.content)
        assert len(data) == 2

    @pytest.mark.django_db
    def test_get_list_prompts_name_filter(self, request_factory):
        target_prompt = baker.make(
            Prompt,
            name="target_prompt",
            prompt_template={"langchain_type": "ChatPromptTemplate"},
            is_active=True,
        )
        baker.make(
            Prompt,
            name="other_prompt",
            prompt_template={"langchain_type": "ChatPromptTemplate"},
            is_active=True,
        )

        request = request_factory.get("/prompts/?name=target_prompt")
        request.json_body = {}

        view = PromptListCreateView()
        response = view.get(request)

        data = json.loads(response.content)
        assert len(data) == 1
        assert data[0]["name"] == target_prompt.name

    @pytest.mark.django_db
    @patch("django_chain.models.Prompt.create_new_version")
    def test_post_create_prompt_success(self, mock_create, request_factory):
        mock_prompt = Mock()
        mock_prompt.to_dict.return_value = {"id": str(uuid.uuid4()), "name": "test_prompt"}
        mock_create.return_value = mock_prompt

        request_data = {
            "name": "test_prompt",
            "prompt_template": {"langchain_type": "ChatPromptTemplate"},
            "input_variables": ["name"],
            "is_active": True,
        }

        request = request_factory.post(
            "/prompts/", data=json.dumps(request_data), content_type="application/json"
        )
        request.json_body = request_data

        view = PromptListCreateView()
        response = view.post(request)

        assert response.status_code == 201
        assert "test_prompt" in str(response.content)

    @pytest.mark.django_db
    @patch("django_chain.models.Prompt.create_new_version")
    def test_post_create_prompt_validation_error(self, mock_create, request_factory):
        mock_create.side_effect = ValidationError({"name": ["This field is required."]})

        request_data = {"name": "", "prompt_template": "Hello"}
        request = request_factory.post(
            "/prompts/", data=json.dumps(request_data), content_type="application/json"
        )
        request.json_body = request_data

        view = PromptListCreateView()
        response = view.post(request)

        assert response.status_code == 400

    @pytest.mark.django_db
    @patch("django_chain.models.Prompt.create_new_version")
    def test_post_create_prompt_generic_exception(self, mock_create, request_factory):
        mock_create.side_effect = Exception("Database error")

        request_data = {"name": "test", "prompt_template": "Hello"}
        request = request_factory.post(
            "/prompts/", data=json.dumps(request_data), content_type="application/json"
        )
        request.json_body = request_data

        view = PromptListCreateView()
        response = view.post(request)

        assert response.status_code == 500


class TestPromptDetailView:
    @pytest.mark.django_db
    def test_get_prompt_success(self, request_factory, sample_prompt):
        request = request_factory.get(f"/prompts/{sample_prompt.id}/")

        view = PromptDetailView()
        response = view.get(request, pk=str(sample_prompt.id))

        assert response.status_code == 200
        data = json.loads(response.content)
        assert data["name"] == sample_prompt.name

    @pytest.mark.django_db
    def test_get_prompt_not_found(self, request_factory):
        non_existent_id = str(uuid.uuid4())
        request = request_factory.get(f"/prompts/{non_existent_id}/")

        view = PromptDetailView()
        response = view.get(request, pk=non_existent_id)

        assert response.status_code == 404

    @pytest.mark.django_db
    def test_put_update_prompt_success(self, request_factory, sample_prompt):
        request_data = {
            "input_variables": ["new_var"],
        }

        request = request_factory.put(
            f"/prompts/{sample_prompt.id}/",
            data=json.dumps(request_data),
            content_type="application/json",
        )
        request.json_body = request_data

        view = PromptDetailView()
        response = view.put(request, pk=str(sample_prompt.id))

        assert response.status_code == 200
        sample_prompt.refresh_from_db()
        assert sample_prompt.input_variables == ["new_var"]

    @pytest.mark.django_db
    def test_put_update_prompt_validation_error(self, request_factory, sample_prompt):
        request_data = {"prompt_template": ""}  # Invalid empty template

        request = request_factory.put(
            f"/prompts/{sample_prompt.id}/",
            data=json.dumps(request_data),
            content_type="application/json",
        )
        request.json_body = request_data

        with patch.object(
            sample_prompt,
            "full_clean",
            side_effect=ValidationError({"prompt_template": ["This field cannot be blank."]}),
        ):
            view = PromptDetailView()
            response = view.put(request, pk=str(sample_prompt.id))

            assert response.status_code == 400

    @pytest.mark.django_db
    def test_delete_prompt_success(self, request_factory, sample_prompt):
        request = request_factory.delete(f"/prompts/{sample_prompt.id}/")

        view = PromptDetailView()
        response = view.delete(request, pk=str(sample_prompt.id))

        assert response.status_code == 204

    @pytest.mark.django_db
    def test_delete_prompt_not_found(self, request_factory):
        non_existent_id = str(uuid.uuid4())
        request = request_factory.delete(f"/prompts/{non_existent_id}/")

        view = PromptDetailView()
        response = view.delete(request, pk=non_existent_id)

        assert response.status_code == 404


class TestPromptActivateDeactivateView:
    @pytest.mark.django_db
    def test_activate_prompt(self, request_factory, sample_prompt):
        sample_prompt.is_active = False
        sample_prompt.save()

        request = request_factory.post(f"/prompts/{sample_prompt.id}/activate/")
        request.json_body = {}

        view = PromptActivateDeactivateView()
        response = view.post(request, pk=str(sample_prompt.id), action="activate")

        assert response.status_code == 200
        sample_prompt.refresh_from_db()
        assert sample_prompt.is_active is True


class TestWorkflowListCreateView:
    @pytest.mark.django_db
    def test_get_list_workflows_success(self, request_factory, sample_workflow):
        request = request_factory.get("/workflows/")
        request.json_body = {}

        view = WorkflowListCreateView()
        response = view.get(request)

        assert response.status_code == 200
        data = json.loads(response.content)
        assert len(data) >= 1
        assert data[0]["name"] == sample_workflow.name

    @pytest.mark.django_db
    def test_post_create_workflow_success(self, request_factory):
        request_data = {
            "name": "test_workflow",
            "description": "Test description",
            "workflow_definition": [
                {"type": "llm"},
                {"type": "parser", "parser_type": "StrOutputParser"},
            ],
            "is_active": False,
        }

        request = request_factory.post(
            "/workflows/", data=json.dumps(request_data), content_type="application/json"
        )
        request.json_body = request_data

        view = WorkflowListCreateView()
        response = view.post(request)

        assert response.status_code == 201
        data = json.loads(response.content)
        assert data["name"] == "test_workflow"

    @pytest.mark.django_db
    def test_post_create_workflow_missing_required_fields(self, request_factory):
        request_data = {"name": "test_workflow"}  # Missing workflow_definition

        request = request_factory.post(
            "/workflows/", data=json.dumps(request_data), content_type="application/json"
        )
        request.json_body = request_data

        view = WorkflowListCreateView()
        response = view.post(request)

        assert response.status_code == 400

    @pytest.mark.django_db
    def test_post_create_workflow_validation_error(self, request_factory):
        request_data = {
            "name": "test_workflow",
            "workflow_definition": {"steps": []},
        }

        request = request_factory.post(
            "/workflows/", data=json.dumps(request_data), content_type="application/json"
        )
        request.json_body = request_data

        with patch(
            "django_chain.models.Workflow.full_clean",
            side_effect=ValidationError({"workflow_definition": ["Invalid definition."]}),
        ):
            view = WorkflowListCreateView()
            response = view.post(request)

            assert response.status_code == 400


class TestWorkflowDetailView:
    @pytest.mark.django_db
    def test_get_workflow_success(self, request_factory, sample_workflow):
        request = request_factory.get(f"/workflows/{sample_workflow.id}/")

        view = WorkflowDetailView()
        response = view.get(request, pk=str(sample_workflow.id))

        assert response.status_code == 200
        data = json.loads(response.content)
        assert data["name"] == sample_workflow.name

    @pytest.mark.django_db
    def test_put_update_workflow_success(self, request_factory, sample_workflow):
        request_data = {
            "description": "Updated description",
            "workflow_definition": [
                {"type": "prompt", "name": "SimpleGreetingPrompt"},
                {"type": "llm", "config": {"temperature": 0.4}},
                {"type": "parser", "parser_type": "StrOutputParser"},
            ],
        }

        request = request_factory.put(
            f"/workflows/{sample_workflow.id}/",
            data=json.dumps(request_data),
            content_type="application/json",
        )
        request.json_body = request_data

        view = WorkflowDetailView()
        response = view.put(request, pk=str(sample_workflow.id))

        assert response.status_code == 200
        sample_workflow.refresh_from_db()
        assert sample_workflow.description == "Updated description"

    @pytest.mark.django_db
    def test_delete_workflow_success(self, request_factory, sample_workflow):
        request = request_factory.delete(f"/workflows/{sample_workflow.id}/")

        view = WorkflowDetailView()
        response = view.delete(request, pk=str(sample_workflow.id))

        assert response.status_code == 204


class TestInteractionLogDetailView:
    @pytest.mark.django_db
    def test_get_interactionlog_success(self, request_factory, sample_interactionlog):
        request = request_factory.get(f"/logs/{sample_interactionlog.id}/")

        view = InteractionLogDetailView()
        response = view.get(request, pk=str(sample_interactionlog.id))

        data = json.loads(response.content)
        assert response.status_code == 200
        assert data["model_name"] == sample_interactionlog.model_name

    @pytest.mark.django_db
    def test_get_interactionlog_fail(self, request_factory, sample_interactionlog):
        request = request_factory.get(f"/logs/")

        view = InteractionLogDetailView()
        response = view.get(request, pk=str(23))

        assert response.status_code == 404

    @pytest.mark.django_db
    def test_delete_interactionlog_success(self, request_factory, sample_interactionlog):
        request = request_factory.delete(f"/logs/")

        view = InteractionLogDetailView()
        response = view.delete(request, pk=str(sample_interactionlog.id))

        assert response.status_code == 204

    @pytest.mark.django_db
    def test_delete_interactionlog_fail(self, request_factory, sample_interactionlog):
        request = request_factory.delete(f"/logs/")

        view = InteractionLogDetailView()
        response = view.delete(request, pk=str(43))

        assert response.status_code == 404


class TestInteractionLogListCreateView:
    @pytest.mark.django_db
    def test_get_list_interactionlog_success(self, request_factory, sample_workflow):
        log = baker.make(InteractionLog)
        request = request_factory.get("/logs/")
        request.json_body = {}

        view = InteractionLogListCreateView()
        response = view.get(request)

        data = json.loads(response.content)
        assert response.status_code == 200
        assert len(data) >= 1


@pytest.mark.skip()
class TestUserInteractionListView:
    @pytest.mark.django_db
    def test_get_list_user_interactions_success(self, request_factory, sample_user_interaction):
        request = request_factory.get("/interactions/")
        request.json_body = {}

        view = UserInteractionListView()
        response = view.get(request)

        assert response.status_code == 200
        data = json.loads(response.content)
        assert len(data) >= 1

    @pytest.mark.django_db
    @pytest.mark.parametrize(
        "filter_param,filter_value",
        [
            ("workflow_name", "test_workflow"),
            ("user_identifier", "test@example.com"),
            ("status", "completed"),
        ],
    )
    def test_get_list_user_interactions_filters(
        self, request_factory, sample_user_interaction, filter_param, filter_value
    ):
        if filter_param == "workflow_name":
            sample_user_interaction.workflow.name = filter_value
            sample_user_interaction.workflow.save()
        elif filter_param == "user_identifier":
            sample_user_interaction.user_identifier = filter_value
            sample_user_interaction.save()
        elif filter_param == "status":
            sample_user_interaction.status = filter_value
            sample_user_interaction.save()

        request = request_factory.get(f"/interactions/?{filter_param}={filter_value}")
        request.json_body = {}

        view = UserInteractionListView()
        response = view.get(request)

        assert response.status_code == 200
        data = json.loads(response.content)
        assert len(data) >= 1

    @pytest.mark.django_db
    def test_get_list_user_interactions_invalid_uuid_filter(self, request_factory):
        request = request_factory.get("/interactions/?workflow_id=invalid-uuid")
        request.json_body = {}

        view = UserInteractionListView()

        with pytest.raises(ValidationError):
            view.get(request)


@pytest.mark.skip()
class TestUserInteractionDetailView:
    @pytest.mark.django_db
    def test_get_user_interaction_success(self, request_factory, sample_user_interaction):
        request = request_factory.get(f"/interactions/{sample_user_interaction.id}/")

        view = UserInteractionDetailView()
        response = view.get(request, pk=str(sample_user_interaction.id))

        assert response.status_code == 200
        data = json.loads(response.content)
        assert data["user_identifier"] == sample_user_interaction.user_identifier

    @pytest.mark.django_db
    def test_get_user_interaction_not_found(self, request_factory):
        non_existent_id = str(uuid.uuid4())
        request = request_factory.get(f"/interactions/{non_existent_id}/")

        view = UserInteractionDetailView()
        response = view.get(request, pk=non_existent_id)

        assert response.status_code == 404


@pytest.mark.django_db
class TestExecuteWorkflowView(TestCase):
    def setUp(self):
        self.client = self.client_class()
        self.workflow_name = "MyTestWorkflow"
        self.url = reverse("execute-workflow", args=[self.workflow_name])

        self.active_workflow = baker.make(
            Workflow,
            name=self.workflow_name,
            workflow_definition=[
                {"type": "prompt", "name": "SimpleGreetingPrompt"},
                {"type": "llm", "config": {"temperature": 0.4}},
                {"type": "parser", "parser_type": "StrOutputParser"},
            ],
            is_active=True,
        )

        self.inactive_workflow = baker.make(
            Workflow,
            name="InactiveWorkflow",
            workflow_definition=[
                {"type": "prompt", "name": "SimpleGreetingPrompt"},
                {"type": "llm", "config": {"temperature": 0.4}},
                {"type": "parser", "parser_type": "StrOutputParser"},
            ],
            is_active=False,
        )

    @patch("django_chain.views._execute_and_log_workflow_step")
    @patch("django_chain.views._to_serializable")
    @patch("django_chain.models.Workflow.to_langchain_chain")
    @pytest.mark.django_db
    def test_post_success(
        self,
        mock_to_langchain,
        mock_to_serializable,
        mock_execute_and_log_workflow_step,
    ):
        """
        Test successful execution of a workflow.
        """
        mock_execute_and_log_workflow_step.return_value = {"result": "success"}
        mock_to_serializable.side_effect = lambda x: x
        mock_to_langchain.return_value = "langchain_object"

        input_data = {"user_input": "hello"}
        response = self.client.post(
            self.url,
            data=json.dumps({"input": input_data}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        response_data = response.json()

        self.assertEqual(response_data["workflow_name"], self.workflow_name)
        self.assertEqual(response_data["input_received"], input_data)
        self.assertEqual(response_data["output"], {"result": "success"})

        mock_execute_and_log_workflow_step.assert_called_once()
        mock_to_serializable.assert_called_once_with({"result": "success"})

    @pytest.mark.django_db
    def test_post_workflow_not_found(self):
        """
        Test case where no active workflow is found with the given name.
        """
        self.active_workflow.delete()
        response = self.client.post(
            self.url,
            data=json.dumps({"input": {}}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 404)
        response_data = response.json()
        self.assertIn("error", response_data)
        self.assertIn(
            f"No active workflow found with name: {self.workflow_name}", response_data["error"]
        )

    @pytest.mark.django_db
    def test_post_invalid_json(self):
        """
        Test case with malformed JSON in the request body.
        """
        response = self.client.post(
            self.url, data="this is not json", content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        response_data = response.json()
        self.assertIn("error", response_data)
        self.assertEqual(response_data["error"], "Invalid JSON in request body.")

    @pytest.mark.django_db
    def test_post_input_data_not_dict(self):
        """
        Test case where 'input' field is not a dictionary.
        """
        response = self.client.post(
            self.url,
            data=json.dumps({"input": "not_a_dict"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        response_data = response.json()
        self.assertIn("error", response_data)

    @patch("django_chain.views._execute_and_log_workflow_step")
    @patch("django_chain.models.Workflow.to_langchain_chain")
    @pytest.mark.django_db
    def test_post_validation_error(self, mock_to_langchain, mock_execute_and_log_workflow_step):
        """
        Test case where a ValidationError is raised during workflow execution.
        """
        mock_execute_and_log_workflow_step.side_effect = ValidationError(
            {"field": ["This field is required."]}
        )
        mock_to_langchain.return_value = "langchain_object"

        response = self.client.post(
            self.url,
            data=json.dumps({"input": {"some_field": "value"}}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        response_data = response.json()
        self.assertIn("error", response_data)
        assert "This field is required." in response_data["error"]

    @patch("django_chain.views._execute_and_log_workflow_step")
    @pytest.mark.django_db
    def test_post_unexpected_exception(self, mock_execute_and_log_workflow_step):
        """
        Test case for an unhandled exception during workflow execution.
        """
        mock_execute_and_log_workflow_step.side_effect = Exception("Something went wrong!")

        response = self.client.post(
            self.url,
            data=json.dumps({"input": {"test": "data"}}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 500)
        response_data = response.json()
        self.assertIn("error", response_data)

    @patch("django_chain.views._execute_and_log_workflow_step")
    @patch("django_chain.views._to_serializable")
    @patch("django_chain.models.Workflow.to_langchain_chain")
    @pytest.mark.django_db
    @override_settings(DJANGO_LLM_SETTINGS={"llm_key": "some_api_key"})
    def test_post_with_global_llm_config(
        self,
        mock_to_langchain,
        mock_to_serializable,
        mock_execute_and_log_workflow_step,
    ):
        """
        Test that global_llm_config from settings is passed correctly.
        """
        mock_execute_and_log_workflow_step.return_value = {"result": "success"}
        mock_to_serializable.side_effect = lambda x: x
        mock_to_langchain.return_value = "langchain_object"

        input_data = {"user_input": "hello"}
        response = self.client.post(
            self.url,
            data=json.dumps({"input": input_data}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

    @patch("django_chain.views._execute_and_log_workflow_step")
    @patch("django_chain.views._to_serializable")
    @patch("django_chain.models.Workflow.to_langchain_chain")
    @pytest.mark.django_db
    def test_post_with_execution_method_kwarg(
        self,
        mock_to_langchain,
        mock_to_serializable,
        mock_execute_and_log_workflow_step,
    ):
        """
        Test that execution_method from kwargs is passed correctly.
        """
        mock_execute_and_log_workflow_step.return_value = {"result": "success"}
        mock_to_serializable.side_effect = lambda x: x
        mock_to_langchain.return_value = "langchain_object"

        input_data = {"user_input": "hello"}

        # Scenario 1: Default execution_method
        response = self.client.post(
            self.url,
            data=json.dumps({"input": input_data}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        mock_execute_and_log_workflow_step.reset_mock()
        pass


@pytest.mark.skip()
@pytest.mark.django_db
class TestChatView:
    @patch("django_chain.views.LLMClient")
    def test_chat_view_success(self, mock_llm_client, request_factory):
        mock_client_instance = Mock()
        mock_client_instance.chat.return_value = {"response": "Hello there!"}
        mock_llm_client.return_value = mock_client_instance

        request_data = {"message": "Hello", "session_id": str(uuid.uuid4())}
        request = request_factory.post(
            "/chat/", data=json.dumps(request_data), content_type="application/json"
        )
        request.body = json.dumps(request_data).encode()

        response = chat_view(request)

        assert response.status_code == 200
        data = json.loads(response.content)
        assert data["response"] == "Hello there!"

    def test_chat_view_missing_message(self, request_factory):
        request_data = {"session_id": str(uuid.uuid4())}
        request = request_factory.post(
            "/chat/", data=json.dumps(request_data), content_type="application/json"
        )
        request.body = json.dumps(request_data).encode()

        response = chat_view(request)

        assert response.status_code == 400
        data = json.loads(response.content)
        assert "error" in data

    @patch("django_chain.views.LLMClient")
    def test_chat_view_exception(self, mock_llm_client, request_factory):
        mock_llm_client.side_effect = Exception("LLM service error")

        request_data = {"message": "Hello"}
        request = request_factory.post(
            "/chat/", data=json.dumps(request_data), content_type="application/json"
        )
        request.body = json.dumps(request_data).encode()

        response = chat_view(request)

        assert response.status_code == 500
        data = json.loads(response.content)
        assert "error" in data


@pytest.mark.skip()
@pytest.mark.django_db
class TestVectorSearchView:
    @patch("django_chain.views.VectorStoreManager")
    def test_vector_search_success(self, mock_vector_manager, request_factory):
        mock_manager_instance = Mock()
        mock_manager_instance.retrieve_documents.return_value = [
            {"doc": "result1"},
            {"doc": "result2"},
        ]
        mock_vector_manager.return_value = mock_manager_instance

        request_data = {"query": "test query", "k": 3}
        request = request_factory.post(
            "/vector-search/", data=json.dumps(request_data), content_type="application/json"
        )
        request.body = json.dumps(request_data).encode()

        response = vector_search_view(request)

        assert response.status_code == 200
        data = json.loads(response.content)
        assert len(data["results"]) == 2

    def test_vector_search_missing_query(self, request_factory):
        request_data = {"k": 5}
        request = request_factory.post(
            "/vector-search/", data=json.dumps(request_data), content_type="application/json"
        )
        request.body = json.dumps(request_data).encode()

        response = vector_search_view(request)

        assert response.status_code == 400
        data = json.loads(response.content)
        assert "error" in data

    @patch("django_chain.views.VectorStoreManager")
    def test_vector_search_exception(self, mock_vector_manager, request_factory):
        mock_vector_manager.side_effect = Exception("Vector store error")

        request_data = {"query": "test query"}
        request = request_factory.post(
            "/vector-search/", data=json.dumps(request_data), content_type="application/json"
        )
        request.body = json.dumps(request_data).encode()

        response = vector_search_view(request)

        assert response.status_code == 500
        data = json.loads(response.content)
        assert "error" in data


class TestUtilityFunctions:
    @pytest.mark.django_db
    def test_serialize_queryset_empty(self):
        result = serialize_queryset([])
        assert result == []

    @pytest.mark.django_db
    def test_serialize_queryset_with_data(self, sample_prompt):
        queryset = [sample_prompt]
        with patch.object(sample_prompt, "to_dict", return_value={"id": "123", "name": "test"}):
            result = serialize_queryset(queryset)
            assert len(result) == 1
            assert result[0]["name"] == "test"

    @pytest.mark.django_db
    def test_serialize_user_interaction_without_logs(self, sample_user_interaction):
        result = serialize_user_interaction(sample_user_interaction)

        assert result["id"] == str(sample_user_interaction.id)
        assert result["user_identifier"] == sample_user_interaction.user_identifier
        assert "interaction_logs" not in result


# Integration tests
@pytest.mark.skip()
@pytest.mark.django_db
class TestViewIntegration:
    def test_prompt_lifecycle(self, request_factory):
        """Test the complete lifecycle of a prompt."""
        # Create
        create_data = {
            "name": "test_prompt_lifecycle",
            "prompt_template": {
                "langchain_type": "ChatPromptTemplate",
                "messages": [{"message_type": "human", "template": "Hello {name}"}],
            },
            "input_variables": ["name"],
            "is_active": True,
        }

        create_request = request_factory.post(
            "/prompts/", data=json.dumps(create_data), content_type="application/json"
        )
        create_request.json_body = create_data

        create_view = PromptListCreateView()

        with patch("django_chain.models.Prompt.create_new_version") as mock_create:
            mock_prompt = baker.make(Prompt, **create_data)
            mock_create.return_value = mock_prompt

            create_response = create_view.post(create_request)
            assert create_response.status_code == 201

            # Get
            get_request = request_factory.get(f"/prompts/{mock_prompt.id}/")
            detail_view = PromptDetailView()
            get_response = detail_view.get(get_request, pk=str(mock_prompt.id))
            assert get_response.status_code == 200

            # Update
            update_data = {"prompt_template": "Updated template"}
            update_request = request_factory.put(
                f"/prompts/{mock_prompt.id}/",
                data=json.dumps(update_data),
                content_type="application/json",
            )
            update_request.json_body = update_data

            update_response = detail_view.put(update_request, pk=str(mock_prompt.id))
            assert update_response.status_code == 200

            # Delete
            delete_request = request_factory.delete(f"/prompts/{mock_prompt.id}/")
            delete_response = detail_view.delete(delete_request, pk=str(mock_prompt.id))
            assert delete_response.status_code == 204
